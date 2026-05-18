"""PostgreSQL helpers and repositories for catalog storage."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import uuid as _uuid
from typing import Any, Callable

try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
except ModuleNotFoundError:  # pragma: no cover - exercised indirectly in tests
    psycopg = None
    dict_row = None

    class _FallbackIdentifier:
        def __init__(self, value: str) -> None:
            self.value = value

        def as_string(self, _context: Any) -> str:
            escaped = self.value.replace('"', '""')
            return f'"{escaped}"'

    class _FallbackSQL:
        def __init__(self, template: str) -> None:
            self.template = template

        def format(self, **kwargs: Any) -> "_FallbackComposed":
            rendered = self.template.format(
                **{
                    key: value.as_string(None) if hasattr(value, "as_string") else str(value)
                    for key, value in kwargs.items()
                }
            )
            return _FallbackComposed(rendered)

    class _FallbackComposed:
        def __init__(self, rendered: str) -> None:
            self.rendered = rendered

        def as_string(self, _context: Any) -> str:
            return self.rendered

    class _FallbackSQLModule:
        Composed = _FallbackComposed

        @staticmethod
        def Identifier(value: str) -> _FallbackIdentifier:
            return _FallbackIdentifier(value)

        @staticmethod
        def SQL(template: str) -> _FallbackSQL:
            return _FallbackSQL(template)

    sql = _FallbackSQLModule()

from config import Settings, get_settings


_IDENTIFIER_MAX_LEN = 63


CATALOG_SCHEMA_DDL = """CREATE SCHEMA IF NOT EXISTS {schema};

CREATE TABLE IF NOT EXISTS {schema}.books (
    book_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL DEFAULT 'bootstrap-owner',
    title TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    added_at TIMESTAMPTZ NOT NULL,
    cover_path TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'unread',
    progress DOUBLE PRECISION NOT NULL DEFAULT 0.0
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_books_owner_source
    ON {schema}.books(owner_user_id, source);

CREATE TABLE IF NOT EXISTS {schema}.notes (
    note_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL DEFAULT 'bootstrap-owner',
    book_id TEXT NOT NULL REFERENCES {schema}.books(book_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_notes_owner_book
    ON {schema}.notes(owner_user_id, book_id);

CREATE TABLE IF NOT EXISTS {schema}.users (
    user_id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'member')),
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS {schema}.conversations (
    conversation_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL REFERENCES {schema}.users(user_id) ON DELETE CASCADE,
    book_id TEXT NOT NULL REFERENCES {schema}.books(book_id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_owner_book_updated
    ON {schema}.conversations(owner_user_id, book_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS {schema}.audit_logs (
    log_id TEXT PRIMARY KEY,
    actor_user_id TEXT NOT NULL,
    actor_role TEXT NOT NULL,
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    result TEXT NOT NULL,
    ip TEXT NOT NULL DEFAULT '',
    user_agent TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL
);
"""


@dataclass(frozen=True)
class PostgresCatalogConfig:
    dsn: str
    schema: str


ConnectionFactory = Callable[[], Any]


def _normalize_schema_name(schema: str) -> str:
    normalized = (schema or "").strip()
    if not normalized:
        raise ValueError("PostgreSQL catalog schema must not be empty.")
    if len(normalized) > _IDENTIFIER_MAX_LEN:
        raise ValueError("PostgreSQL catalog schema exceeds identifier length limit.")
    return normalized


def build_postgres_dsn(settings: Settings | None = None) -> str:
    current = settings or get_settings()
    if current.postgres_dsn:
        return current.postgres_dsn
    return (
        f"postgresql://{current.postgres_user}:{current.postgres_password}"
        f"@{current.postgres_host}:{current.postgres_port}/{current.postgres_database}"
    )


def get_postgres_catalog_config(settings: Settings | None = None) -> PostgresCatalogConfig:
    current = settings or get_settings()
    return PostgresCatalogConfig(
        dsn=build_postgres_dsn(current),
        schema=_normalize_schema_name(current.postgres_catalog_schema),
    )


def get_postgres_connection(settings: Settings | None = None) -> Any:
    if psycopg is None:
        raise RuntimeError("psycopg is required for PostgreSQL connections.")
    config = get_postgres_catalog_config(settings)
    return psycopg.connect(
        config.dsn,
        row_factory=dict_row,
        options=f"-c search_path={config.schema}",
    )


def ensure_postgres_schema(settings: Settings | None = None, connection_factory: ConnectionFactory | None = None) -> None:
    conn = (connection_factory or get_postgres_connection)(settings) if connection_factory is None else connection_factory()
    try:
        ddl = render_catalog_schema_ddl(get_postgres_catalog_config(settings).schema).as_string(None)
        cur = conn.cursor()
        for stmt in ddl.split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class _PostgresRepository:
    def __init__(self, connection_factory: ConnectionFactory | None = None) -> None:
        self._connection_factory = connection_factory or get_postgres_connection

    def _connection(self) -> Any:
        return self._connection_factory()

    def _execute_write(self, query: str, params: tuple[Any, ...]) -> None:
        conn = self._connection()
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _fetchone(self, query: str, params: tuple[Any, ...]) -> dict | None:
        conn = self._connection()
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def _fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[dict]:
        conn = self._connection()
        try:
            cur = conn.cursor()
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()


class PostgresBookCatalog(_PostgresRepository):
    def add(
        self,
        *,
        owner_user_id: str,
        book_id: str,
        title: str,
        author: str,
        source: str,
        total_chunks: int,
        cover_path: str = "",
    ) -> None:
        added_at = datetime.now(tz=timezone.utc)
        self._execute_write(
            """
            INSERT INTO books (book_id, owner_user_id, title, author, source, total_chunks, added_at, cover_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (owner_user_id, source) DO UPDATE SET
                title = EXCLUDED.title,
                author = EXCLUDED.author,
                total_chunks = EXCLUDED.total_chunks,
                added_at = EXCLUDED.added_at,
                cover_path = EXCLUDED.cover_path
            """,
            (book_id, owner_user_id, title, author, source, total_chunks, added_at, cover_path),
        )

    def update_progress(self, book_id: str, progress: float, *, owner_user_id: str) -> None:
        status = "completed" if progress >= 1.0 else "reading"
        self._execute_write(
            "UPDATE books SET progress = %s, status = %s WHERE book_id = %s AND owner_user_id = %s",
            (progress, status, book_id, owner_user_id),
        )

    def update_status(self, book_id: str, status: str, *, owner_user_id: str) -> None:
        self._execute_write(
            "UPDATE books SET status = %s WHERE book_id = %s AND owner_user_id = %s",
            (status, book_id, owner_user_id),
        )

    def delete(self, book_id: str, *, owner_user_id: str) -> None:
        self._execute_write("DELETE FROM books WHERE book_id = %s AND owner_user_id = %s", (book_id, owner_user_id))

    def get_all(self, *, owner_user_id: str) -> list[dict]:
        return self._fetchall("SELECT * FROM books WHERE owner_user_id = %s ORDER BY added_at DESC", (owner_user_id,))

    def get_by_id(self, book_id: str, *, owner_user_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM books WHERE book_id = %s AND owner_user_id = %s", (book_id, owner_user_id))

    def get_by_source(self, source: str, *, owner_user_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM books WHERE source = %s AND owner_user_id = %s", (source, owner_user_id))


class PostgresNoteCatalog(_PostgresRepository):
    def upsert(self, *, owner_user_id: str, book_id: str, file_path: str) -> None:
        now = datetime.now(tz=timezone.utc)
        note_id = str(_uuid.uuid4())
        self._execute_write(
            """
            INSERT INTO notes (note_id, owner_user_id, book_id, file_path, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (owner_user_id, book_id) DO UPDATE SET
                file_path = EXCLUDED.file_path,
                updated_at = EXCLUDED.updated_at
            """,
            (note_id, owner_user_id, book_id, file_path, now, now),
        )

    def touch(self, book_id: str, *, owner_user_id: str) -> None:
        now = datetime.now(tz=timezone.utc)
        self._execute_write(
            "UPDATE notes SET updated_at = %s WHERE book_id = %s AND owner_user_id = %s",
            (now, book_id, owner_user_id),
        )

    def get_by_book_id(self, book_id: str, *, owner_user_id: str) -> dict | None:
        return self._fetchone("SELECT * FROM notes WHERE book_id = %s AND owner_user_id = %s", (book_id, owner_user_id))

    def get_all(self, *, owner_user_id: str) -> list[dict]:
        return self._fetchall("SELECT * FROM notes WHERE owner_user_id = %s ORDER BY updated_at DESC", (owner_user_id,))

    def delete(self, book_id: str, *, owner_user_id: str) -> None:
        self._execute_write("DELETE FROM notes WHERE book_id = %s AND owner_user_id = %s", (book_id, owner_user_id))


def render_catalog_schema_ddl(schema: str) -> sql.Composed:
    schema_ident = sql.Identifier(_normalize_schema_name(schema))
    return sql.SQL(CATALOG_SCHEMA_DDL).format(schema=schema_ident)


def get_book_catalog() -> PostgresBookCatalog:
    return PostgresBookCatalog()


def get_note_catalog() -> PostgresNoteCatalog:
    return PostgresNoteCatalog()


__all__ = [
    "CATALOG_SCHEMA_DDL",
    "PostgresBookCatalog",
    "PostgresCatalogConfig",
    "PostgresNoteCatalog",
    "build_postgres_dsn",
    "ensure_postgres_schema",
    "get_book_catalog",
    "get_postgres_catalog_config",
    "get_postgres_connection",
    "get_note_catalog",
    "render_catalog_schema_ddl",
]
