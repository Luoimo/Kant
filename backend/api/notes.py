from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import get_settings
from backend.services.note_service import NoteService
from backend.storage.book_catalog import get_book_catalog
from backend.storage.note_vector_store import make_note_vector_store

router = APIRouter(prefix="/notes", tags=["notes"])


@lru_cache(maxsize=1)
def _service() -> NoteService:
    settings = get_settings()
    return NoteService(
        notes_dir=Path(settings.note_storage_dir),
        note_vector_store=make_note_vector_store(settings),
    )


def _require_book(book_id: str) -> dict:
    entry = get_book_catalog().get_by_id(book_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"书籍不存在：{book_id}")
    return entry


class AppendNoteRequest(BaseModel):
    book_id: str
    content: str


@router.get("/books")
def list_note_books() -> list[dict]:
    """返回已有笔记的书籍列表。"""
    return _service().list_books()


@router.get("/timeline")
def get_timeline(book_id: str | None = None) -> dict:
    """返回结构化元数据供前端时间轴可视化。可按 book_id 过滤。"""
    if book_id:
        _require_book(book_id)
    return _service().get_timeline(book_id)


@router.get("/{book_id}")
def get_note(book_id: str) -> dict:
    """返回指定书籍的笔记全文，供前端编辑器展示。"""
    book = _require_book(book_id)
    content = _service().get_note_content(book_id)
    return {"book_id": book_id, "book_title": book["title"], "content": content}


@router.post("/append")
def append_note(req: AppendNoteRequest) -> dict:
    """追加用户手写笔记到指定书籍的笔记文件。"""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="content 不能为空")
    book = _require_book(req.book_id)
    _service().append_manual(req.content, req.book_id)
    return {"ok": True, "book_id": req.book_id, "book_title": book["title"]}
