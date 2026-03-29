from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.storage.book_catalog import get_book_catalog

router = APIRouter(prefix="/reader", tags=["reader"])


def _resolve(book_id: str) -> dict:
    """Return book catalog entry, raise 404 if not found."""
    entry = get_book_catalog().get_by_id(book_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"书籍不存在：{book_id}")
    return entry


@router.post("/{book_id}/open")
def reader_open(book_id: str) -> dict:
    """Mark book as reading when user opens it."""
    book = _resolve(book_id)
    get_book_catalog().update_status(book_id, "reading")
    return {"book_id": book_id, "book_title": book["title"]}
