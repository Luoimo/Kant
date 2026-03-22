from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import get_settings
from backend.rag.chroma.chroma_store import ChromaStore
from backend.services.plan_generator import PlanGenerator
from backend.storage.plan_storage import LocalPlanStorage, safe_plan_name

router = APIRouter(prefix="/reader", tags=["reader"])


def _plan_storage() -> LocalPlanStorage:
    return LocalPlanStorage(Path(get_settings().plan_storage_dir))


def _resolve(book_id: str) -> tuple[str, str]:
    """Return (book_title, book_source) for a book_id, raise 404 if not found.

    Note: ChromaStore() is instantiated per request. For a small personal
    library this is acceptable. If latency becomes a concern, replace with
    a module-level singleton.
    """
    entry = ChromaStore().resolve_book_by_id(book_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"书籍不存在：{book_id}")
    return entry["book_title"], entry["source"]


class ReaderInitRequest(BaseModel):
    reading_goal: str = ""


class ReaderProgressRequest(BaseModel):
    chapter: str


@router.post("/{book_id}/init")
def reader_init(book_id: str, req: ReaderInitRequest) -> dict:
    """Auto-generate a reading plan when user opens a book. Idempotent."""
    book_title, book_source = _resolve(book_id)
    gen = PlanGenerator()
    plan = gen.generate(book_title, book_source=book_source, reading_goal=req.reading_goal)
    return {"book_id": book_id, "book_title": book_title, "plan": plan}


@router.get("/{book_id}/plan")
def reader_get_plan(book_id: str) -> dict:
    """Return the current plan markdown for sidebar display."""
    book_title, _ = _resolve(book_id)
    path = Path(get_settings().plan_storage_dir) / f"{safe_plan_name(book_title)}.md"
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    return {"book_id": book_id, "book_title": book_title, "plan": content}


@router.post("/{book_id}/progress")
def reader_progress(book_id: str, req: ReaderProgressRequest) -> dict:
    """Mark a chapter as complete (toggles checkbox in plan file)."""
    book_title, _ = _resolve(book_id)
    storage = _plan_storage()
    if not storage.find_by_book(book_title):
        raise HTTPException(status_code=404, detail="该书暂无阅读计划，请先初始化")
    if not storage.mark_chapter_done(book_title, req.chapter):
        raise HTTPException(status_code=404, detail=f"未找到章节：{req.chapter}")
    return {"ok": True, "book_id": book_id, "book_title": book_title, "chapter": req.chapter}
