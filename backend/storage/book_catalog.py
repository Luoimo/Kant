"""SQLite-backed catalog for books, notes, and plans.

All three share one DB file (data/books.db).
File I/O stays in the agent/storage layer; this module only tracks metadata.
"""
from __future__ import annotations

import sqlite3
import uuid as _uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def book_id_from_source(source: str) -> str:
    """Deterministic book_id: uuid5(NAMESPACE_URL, source). Same formula as chroma_store.ingest()."""
    return str(_uuid.uuid5(_uuid.NAMESPACE_URL, source))


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS books (
    book_id      TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    author       TEXT NOT NULL DEFAULT '',
    source       TEXT NOT NULL UNIQUE,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    added_at     TEXT NOT NULL,
    cover_path   TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'unread',
    progress     REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS notes (
    note_id    TEXT PRIMARY KEY,
    book_id    TEXT NOT NULL UNIQUE,
    file_path  TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plans (
    plan_id      TEXT PRIMARY KEY,
    book_id      TEXT NOT NULL UNIQUE,
    file_path    TEXT NOT NULL,
    reading_goal TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Shared connection mixin
# ---------------------------------------------------------------------------

class _DB:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            for stmt in _DDL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# BookCatalog
# ---------------------------------------------------------------------------

class BookCatalog(_DB):

    def add(self, *, book_id: str, title: str, author: str, source: str,
            total_chunks: int, cover_path: str = "") -> None:
        added_at = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO books (book_id, title, author, source, total_chunks, added_at, cover_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    title        = excluded.title,
                    author       = excluded.author,
                    total_chunks = excluded.total_chunks,
                    added_at     = excluded.added_at,
                    cover_path   = excluded.cover_path
                """,
                (book_id, title, author, source, total_chunks, added_at, cover_path),
            )

    def update_progress(self, book_id: str, progress: float) -> None:
        status = "completed" if progress >= 1.0 else "reading"
        with self._connect() as conn:
            conn.execute(
                "UPDATE books SET progress = ?, status = ? WHERE book_id = ?",
                (progress, status, book_id),
            )

    def update_status(self, book_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE books SET status = ? WHERE book_id = ?", (status, book_id))

    def delete(self, book_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM books WHERE book_id = ?", (book_id,))

    def get_all(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM books ORDER BY added_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_by_id(self, book_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM books WHERE book_id = ?", (book_id,)).fetchone()
        return dict(row) if row else None

    def get_by_source(self, source: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM books WHERE source = ?", (source,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# NoteCatalog
# ---------------------------------------------------------------------------

class NoteCatalog(_DB):
    def upsert(self, *, book_id: str, file_path: str) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        note_id = str(_uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO notes (note_id, book_id, file_path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(book_id) DO UPDATE SET
                    file_path  = excluded.file_path,
                    updated_at = excluded.updated_at
                """,
                (note_id, book_id, file_path, now, now),
            )

    def touch(self, book_id: str) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("UPDATE notes SET updated_at = ? WHERE book_id = ?", (now, book_id))

    def get_by_book_id(self, book_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM notes WHERE book_id = ?", (book_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_all(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM notes ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]

    def delete(self, book_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM notes WHERE book_id = ?", (book_id,))


# ---------------------------------------------------------------------------
# PlanCatalog
# ---------------------------------------------------------------------------

class PlanCatalog(_DB):
    def upsert(self, *, book_id: str, file_path: str, reading_goal: str = "") -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        plan_id = str(_uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO plans (plan_id, book_id, file_path, reading_goal, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(book_id) DO UPDATE SET
                    file_path    = excluded.file_path,
                    reading_goal = excluded.reading_goal,
                    updated_at   = excluded.updated_at
                """,
                (plan_id, book_id, file_path, reading_goal, now, now),
            )

    def touch(self, book_id: str) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute("UPDATE plans SET updated_at = ? WHERE book_id = ?", (now, book_id))

    def get_by_book_id(self, book_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM plans WHERE book_id = ?", (book_id,)
            ).fetchone()
        return dict(row) if row else None

    def delete(self, book_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM plans WHERE book_id = ?", (book_id,))


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _db_path() -> Path:
    from backend.config import get_settings
    return Path(get_settings().book_catalog_db)


def get_book_catalog() -> BookCatalog:
    return BookCatalog(_db_path())


def get_note_catalog() -> NoteCatalog:
    return NoteCatalog(_db_path())


def get_plan_catalog() -> PlanCatalog:
    return PlanCatalog(_db_path())


__all__ = [
    "BookCatalog", "NoteCatalog", "PlanCatalog",
    "get_book_catalog", "get_note_catalog", "get_plan_catalog",
    "book_id_from_source",
]
