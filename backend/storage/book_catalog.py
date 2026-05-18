"""Catalog compatibility exports backed by PostgreSQL repositories."""
from __future__ import annotations

from storage import postgres


BookCatalog = postgres.PostgresBookCatalog
NoteCatalog = postgres.PostgresNoteCatalog


def get_book_catalog() -> BookCatalog:
    return postgres.get_book_catalog()


def get_note_catalog() -> NoteCatalog:
    return postgres.get_note_catalog()


__all__ = [
    "BookCatalog",
    "NoteCatalog",
    "get_book_catalog",
    "get_note_catalog",
]
