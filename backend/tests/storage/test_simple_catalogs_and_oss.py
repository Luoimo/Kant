from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import storage.audit_log_catalog as audit_module
import storage.conversation_catalog as conversation_module
import storage.oss_client as oss_module
import storage.user_catalog as user_module
from storage.audit_log_catalog import AuditLogCatalog
from storage.conversation_catalog import ConversationCatalog
from storage.oss_client import OSSClient, build_oss_uri, is_oss_uri, parse_oss_uri
from storage.user_catalog import UserCatalog


class Cursor:
    def __init__(self, *, one=None, many=None) -> None:
        self.one = one
        self.many = many or []
        self.calls: list[tuple[str, object]] = []

    def execute(self, query, params=None):
        self.calls.append((str(query), params))

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.many


class Connection:
    def __init__(self, cursor: Cursor) -> None:
        self._cursor = cursor
        self.commits = 0
        self.closed = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed += 1


def test_user_catalog_crud_closes_connections(monkeypatch) -> None:
    created = {"user_id": "u1", "email": "a@example.com", "role": "member"}
    by_email = {**created, "password_hash": "hash"}
    cursors = [
        Cursor(one=created),
        Cursor(one=by_email),
        Cursor(one=created),
        Cursor(one=None),
        Cursor(many=[created]),
    ]
    connections = [Connection(cursor) for cursor in cursors]
    iterator = iter(connections)
    monkeypatch.setattr(user_module, "get_postgres_connection", lambda: next(iterator))
    catalog = UserCatalog()

    assert catalog.create_member(email="a@example.com", password_hash="hash") == created
    assert catalog.get_by_email("a@example.com") == by_email
    assert catalog.get_by_id("u1") == created
    assert catalog.get_by_id("missing") is None
    assert catalog.list_all() == [created]
    assert connections[0].commits == 1
    assert all(connection.closed == 1 for connection in connections)
    assert "INSERT INTO users" in cursors[0].calls[0][0]
    assert cursors[1].calls[0][1] == ("a@example.com",)


def test_conversation_catalog_create_list_and_get(monkeypatch) -> None:
    row = {"conversation_id": "c1", "owner_user_id": "u1", "book_id": "b1", "title": "Chat"}
    cursors = [Cursor(one=row), Cursor(many=[row]), Cursor(one=row), Cursor(one=None)]
    connections = [Connection(cursor) for cursor in cursors]
    iterator = iter(connections)
    monkeypatch.setattr(conversation_module, "get_postgres_connection", lambda: next(iterator))
    catalog = ConversationCatalog()

    assert catalog.create(owner_user_id="u1", book_id="b1", title="Chat") == row
    assert catalog.list_by_book(owner_user_id="u1", book_id="b1") == [row]
    assert catalog.get(owner_user_id="u1", conversation_id="c1") == row
    assert catalog.get(owner_user_id="u1", conversation_id="missing") is None
    assert connections[0].commits == 1
    assert all(connection.closed == 1 for connection in connections)


def test_audit_log_write_and_list(monkeypatch) -> None:
    row = {"log_id": "l1", "action": "login"}
    cursors = [Cursor(), Cursor(many=[row])]
    connections = [Connection(cursor) for cursor in cursors]
    iterator = iter(connections)
    monkeypatch.setattr(audit_module, "get_postgres_connection", lambda: next(iterator))
    catalog = AuditLogCatalog()

    catalog.write(
        actor_user_id="u1",
        actor_role="admin",
        action="login",
        resource_type="session",
        resource_id="s1",
        result="success",
        ip="127.0.0.1",
        user_agent="pytest",
    )
    assert catalog.list_recent(limit=5) == [row]
    assert connections[0].commits == 1
    assert cursors[1].calls[0][1] == (5,)
    assert all(connection.closed == 1 for connection in connections)


def oss_settings(**overrides):
    data = {
        "oss_access_key_id": "id",
        "oss_secret_access_key": "secret",
        "oss_endpoint": "https://oss.example.com",
        "oss_bucket": "kant",
        "oss_books_prefix": "books",
        "oss_covers_prefix": "covers",
        "oss_signed_url_expires": 600,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_oss_uri_helpers() -> None:
    assert is_oss_uri("oss://bucket/key") is True
    assert is_oss_uri(None) is False
    assert build_oss_uri("bucket", "path/file") == "oss://bucket/path/file"
    assert parse_oss_uri("oss://bucket/path/file") == ("bucket", "path/file")
    with pytest.raises(ValueError):
        parse_oss_uri("https://example.com/file")
    with pytest.raises(ValueError):
        parse_oss_uri("oss://bucket")


def test_oss_client_disabled_and_enabled_operations(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        oss_module,
        "get_settings",
        lambda: oss_settings(oss_access_key_id="", oss_secret_access_key="", oss_bucket=""),
    )
    disabled = OSSClient()
    assert disabled.enabled is False
    assert disabled._bucket is None

    bucket = MagicMock()
    bucket.object_exists.side_effect = [True, False, True]
    fake_oss2 = SimpleNamespace(
        Auth=MagicMock(return_value="auth"),
        Bucket=MagicMock(return_value=bucket),
    )
    monkeypatch.setattr(oss_module, "oss2", fake_oss2)
    monkeypatch.setattr(oss_module, "get_settings", lambda: oss_settings())
    client = OSSClient()
    assert client.enabled is True
    assert client.book_key("u1", "book.epub") == "users/u1/books/book.epub"
    assert client.cover_key("u1", "b1", "png") == "users/u1/covers/b1.png"
    assert client.cover_key("u1", "b1", ".jpg") == "users/u1/covers/b1.jpg"

    assert client.put_bytes("key", b"data", "application/epub+zip") == "oss://kant/key"
    assert client.put_file("key", tmp_path / "book.epub") == "oss://kant/key"
    destination = tmp_path / "nested" / "download.epub"
    client.get_to_file("key", destination)
    assert destination.parent.exists()
    assert client.delete("key") is True
    assert client.delete("missing") is False
    assert client.exists("key") is True

    bucket.sign_url.return_value = "https://signed"
    assert client.signed_url("key") == "https://signed"
    assert client.signed_url_from_uri("oss://kant/key", expires=30) == "https://signed"
    with pytest.raises(ValueError):
        client.signed_url_from_uri("oss://other/key")


def test_oss_client_errors_and_cached_factory(monkeypatch) -> None:
    client = object.__new__(OSSClient)
    client.bucket_name = "kant"
    client.signed_url_expires = 60
    client._bucket = MagicMock()
    client._bucket.object_exists.side_effect = RuntimeError("offline")
    assert client.delete("key") is False

    monkeypatch.setattr(oss_module, "get_settings", lambda: oss_settings())
    monkeypatch.setattr(
        oss_module,
        "oss2",
        SimpleNamespace(Auth=MagicMock(return_value="auth"), Bucket=MagicMock(return_value=MagicMock())),
    )
    oss_module.get_oss_client.cache_clear()
    first = oss_module.get_oss_client()
    assert oss_module.get_oss_client() is first
    oss_module.get_oss_client.cache_clear()
