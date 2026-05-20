from __future__ import annotations

from datetime import datetime, timezone

from config import Settings
from storage.book_catalog import BookCatalog, NoteCatalog, get_book_catalog, get_note_catalog
from storage.postgres import (
    PostgresBookCatalog,
    PostgresCatalogConfig,
    PostgresNoteCatalog,
    build_postgres_dsn,
    get_postgres_catalog_config,
    render_catalog_schema_ddl,
)


class FakeCursor:
    def __init__(self, *, fetchone_result=None, fetchall_result=None, fail: Exception | None = None) -> None:
        self.fetchone_result = fetchone_result
        self.fetchall_result = fetchall_result or []
        self.fail = fail
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, query: str, params: tuple = ()) -> None:
        self.executed.append((query, params))
        if self.fail:
            raise self.fail

    def fetchone(self):
        return self.fetchone_result

    def fetchall(self):
        return self.fetchall_result


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.commit_calls = 0
        self.rollback_calls = 0
        self.close_calls = 0

    def cursor(self) -> FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def _last_query(cursor: FakeCursor) -> tuple[str, tuple]:
    assert cursor.executed
    return cursor.executed[-1]


def test_build_postgres_dsn_from_components() -> None:
    settings = Settings(
        postgres_dsn="",
        postgres_host="db.internal",
        postgres_port=5433,
        postgres_user="reader",
        postgres_password="secret",
        postgres_database="kant_dev",
    )

    dsn = build_postgres_dsn(settings)

    assert dsn == "postgresql://reader:secret@db.internal:5433/kant_dev"


def test_build_postgres_dsn_prefers_explicit_value() -> None:
    settings = Settings(
        postgres_dsn="postgresql://service:pass@db.example.com:5432/kant",
        database_url="postgresql://database-url",
        postgres_host="ignored",
        postgres_port=1111,
        postgres_user="ignored",
        postgres_password="ignored",
        postgres_database="ignored",
    )

    dsn = build_postgres_dsn(settings)

    assert dsn == "postgresql://service:pass@db.example.com:5432/kant"


def test_build_postgres_dsn_uses_database_url() -> None:
    settings = Settings(
        postgres_dsn="",
        database_url="postgresql://service:pass@db.internal:5432/kant",
    )

    dsn = build_postgres_dsn(settings)

    assert dsn == "postgresql://service:pass@db.internal:5432/kant"


def test_get_postgres_catalog_config_returns_normalized_schema() -> None:
    settings = Settings(
        postgres_dsn="postgresql://service:pass@db.example.com:5432/kant",
        postgres_catalog_schema="  catalog_app  ",
    )

    config = get_postgres_catalog_config(settings)

    assert config == PostgresCatalogConfig(
        dsn="postgresql://service:pass@db.example.com:5432/kant",
        schema="catalog_app",
    )


def test_render_catalog_schema_ddl_targets_schema() -> None:
    ddl = render_catalog_schema_ddl("catalog")
    text = ddl.as_string(None)
    assert 'CREATE TABLE IF NOT EXISTS "catalog".books' in text
    assert "owner_user_id TEXT NOT NULL" in text
    assert 'CREATE TABLE IF NOT EXISTS "catalog".notes' in text
    assert 'CREATE TABLE IF NOT EXISTS "catalog".users' in text
    assert 'CREATE TABLE IF NOT EXISTS "catalog".conversations' in text
    assert 'CREATE TABLE IF NOT EXISTS "catalog".audit_logs' in text


def test_render_catalog_schema_ddl_rejects_empty_schema() -> None:
    try:
        render_catalog_schema_ddl("   ")
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty schema.")


def test_book_catalog_add_upserts_by_source_and_commits() -> None:
    cursor = FakeCursor()
    conn = FakeConnection(cursor)
    catalog = PostgresBookCatalog(connection_factory=lambda: conn)

    catalog.add(
        owner_user_id="user-1",
        book_id="book-1",
        title="Kant",
        author="Immanuel Kant",
        source="oss://books/kant.epub",
        total_chunks=42,
        cover_path="oss://covers/kant.png",
    )

    query, params = _last_query(cursor)
    assert "INSERT INTO books" in query
    assert "ON CONFLICT (owner_user_id, source) DO UPDATE SET" in query
    assert params[:6] == (
        "book-1",
        "user-1",
        "Kant",
        "Immanuel Kant",
        "oss://books/kant.epub",
        42,
    )
    assert isinstance(params[6], datetime)
    assert params[6].tzinfo == timezone.utc
    assert params[7] == "oss://covers/kant.png"
    assert conn.commit_calls == 1
    assert conn.rollback_calls == 0
    assert conn.close_calls == 1


def test_book_catalog_getters_and_delete_use_postgres_placeholders() -> None:
    row = {
        "book_id": "book-1",
        "title": "Kant",
        "author": "Immanuel Kant",
        "source": "oss://books/kant.epub",
        "total_chunks": 42,
        "added_at": datetime(2026, 5, 13, tzinfo=timezone.utc),
        "cover_path": "oss://covers/kant.png",
        "status": "reading",
        "progress": 0.5,
    }
    get_by_id_cursor = FakeCursor(fetchone_result=row)
    get_all_cursor = FakeCursor(fetchall_result=[row])
    get_by_source_cursor = FakeCursor(fetchone_result=row)
    delete_cursor = FakeCursor()
    connections = iter(
        [
            FakeConnection(get_by_id_cursor),
            FakeConnection(get_all_cursor),
            FakeConnection(get_by_source_cursor),
            FakeConnection(delete_cursor),
        ]
    )
    catalog = PostgresBookCatalog(connection_factory=lambda: next(connections))

    assert catalog.get_by_id("book-1", owner_user_id="user-1") == row
    assert catalog.get_all(owner_user_id="user-1") == [row]
    assert catalog.get_by_source("oss://books/kant.epub", owner_user_id="user-1") == row
    catalog.delete("book-1", owner_user_id="user-1")

    assert _last_query(get_by_id_cursor) == (
        "SELECT * FROM books WHERE book_id = %s AND owner_user_id = %s",
        ("book-1", "user-1"),
    )
    assert _last_query(get_all_cursor) == (
        "SELECT * FROM books WHERE owner_user_id = %s ORDER BY added_at DESC",
        ("user-1",),
    )
    assert _last_query(get_by_source_cursor) == (
        "SELECT * FROM books WHERE source = %s AND owner_user_id = %s",
        ("oss://books/kant.epub", "user-1"),
    )
    assert _last_query(delete_cursor) == (
        "DELETE FROM books WHERE book_id = %s AND owner_user_id = %s",
        ("book-1", "user-1"),
    )


def test_book_catalog_update_progress_sets_completed_status_at_one() -> None:
    cursor = FakeCursor()
    conn = FakeConnection(cursor)
    catalog = PostgresBookCatalog(connection_factory=lambda: conn)

    catalog.update_progress("book-1", 1.0, owner_user_id="user-1")

    assert _last_query(cursor) == (
        "UPDATE books SET progress = %s, status = %s WHERE book_id = %s AND owner_user_id = %s",
        (1.0, "completed", "book-1", "user-1"),
    )
    assert conn.commit_calls == 1


def test_book_catalog_update_status_uses_supplied_status() -> None:
    cursor = FakeCursor()
    conn = FakeConnection(cursor)
    catalog = PostgresBookCatalog(connection_factory=lambda: conn)

    catalog.update_status("book-1", "unread", owner_user_id="user-1")

    assert _last_query(cursor) == (
        "UPDATE books SET status = %s WHERE book_id = %s AND owner_user_id = %s",
        ("unread", "book-1", "user-1"),
    )


def test_book_catalog_rolls_back_and_reraises_on_write_error() -> None:
    cursor = FakeCursor(fail=RuntimeError("boom"))
    conn = FakeConnection(cursor)
    catalog = PostgresBookCatalog(connection_factory=lambda: conn)

    try:
        catalog.delete("book-1", owner_user_id="user-1")
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("Expected write failure to propagate.")

    assert conn.commit_calls == 0
    assert conn.rollback_calls == 1
    assert conn.close_calls == 1


def test_note_catalog_upsert_and_touch_use_postgres_sql() -> None:
    upsert_cursor = FakeCursor()
    touch_cursor = FakeCursor()
    connections = iter([FakeConnection(upsert_cursor), FakeConnection(touch_cursor)])
    catalog = PostgresNoteCatalog(connection_factory=lambda: next(connections))

    catalog.upsert(owner_user_id="user-1", book_id="book-1", file_path="/tmp/kant.md")
    catalog.touch("book-1", owner_user_id="user-1")

    upsert_query, upsert_params = _last_query(upsert_cursor)
    assert "INSERT INTO notes" in upsert_query
    assert "ON CONFLICT (owner_user_id, book_id) DO UPDATE SET" in upsert_query
    assert len(upsert_params) == 6
    assert upsert_params[1] == "user-1"
    assert upsert_params[2] == "book-1"
    assert upsert_params[3] == "/tmp/kant.md"
    assert isinstance(upsert_params[4], datetime)
    assert isinstance(upsert_params[5], datetime)
    assert _last_query(touch_cursor) == (
        "UPDATE notes SET updated_at = %s WHERE book_id = %s AND owner_user_id = %s",
        (touch_cursor.executed[0][1][0], "book-1", "user-1"),
    )


def test_note_catalog_getters_and_delete_return_rows() -> None:
    row = {
        "note_id": "note-1",
        "book_id": "book-1",
        "file_path": "/tmp/kant.md",
        "created_at": datetime(2026, 5, 13, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 5, 13, tzinfo=timezone.utc),
    }
    get_by_book_cursor = FakeCursor(fetchone_result=row)
    get_all_cursor = FakeCursor(fetchall_result=[row])
    delete_cursor = FakeCursor()
    connections = iter(
        [
            FakeConnection(get_by_book_cursor),
            FakeConnection(get_all_cursor),
            FakeConnection(delete_cursor),
        ]
    )
    catalog = PostgresNoteCatalog(connection_factory=lambda: next(connections))

    assert catalog.get_by_book_id("book-1", owner_user_id="user-1") == row
    assert catalog.get_all(owner_user_id="user-1") == [row]
    catalog.delete("book-1", owner_user_id="user-1")

    assert _last_query(get_by_book_cursor) == (
        "SELECT * FROM notes WHERE book_id = %s AND owner_user_id = %s",
        ("book-1", "user-1"),
    )
    assert _last_query(get_all_cursor) == (
        "SELECT * FROM notes WHERE owner_user_id = %s ORDER BY updated_at DESC",
        ("user-1",),
    )
    assert _last_query(delete_cursor) == (
        "DELETE FROM notes WHERE book_id = %s AND owner_user_id = %s",
        ("book-1", "user-1"),
    )


def test_book_catalog_module_compatibility_exports_postgres_repositories(monkeypatch) -> None:
    expected_book = object()
    expected_note = object()

    monkeypatch.setattr("storage.postgres.get_book_catalog", lambda: expected_book)
    monkeypatch.setattr("storage.postgres.get_note_catalog", lambda: expected_note)

    assert BookCatalog is PostgresBookCatalog
    assert NoteCatalog is PostgresNoteCatalog
    assert get_book_catalog() is expected_book
    assert get_note_catalog() is expected_note
