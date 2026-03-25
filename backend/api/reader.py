from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from backend.config import get_settings
from backend.storage.book_catalog import get_book_catalog, get_plan_catalog
from backend.storage.plan_storage import LocalPlanStorage
from backend.team.team import get_plan_editor

router = APIRouter(prefix="/reader", tags=["reader"])


def _resolve(book_id: str) -> dict:
    """Return book catalog entry, raise 404 if not found."""
    entry = get_book_catalog().get_by_id(book_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"书籍不存在：{book_id}")
    return entry


def _compute_progress(plan_path: Path) -> float:
    content = plan_path.read_text(encoding="utf-8")
    total = len(re.findall(r"- \[[ x]\]", content))
    done = len(re.findall(r"- \[x\]", content))
    return done / total if total > 0 else 0.0


def _mark_chapter_done(plan_path: Path, chapter: str) -> bool:
    content = plan_path.read_text(encoding="utf-8")
    escaped = re.escape(chapter.strip())
    updated = re.sub(rf"- \[ \] ({escaped}[^\n]*)", r"- [x] \1", content)
    if updated == content:
        return False
    plan_path.write_text(updated, encoding="utf-8")
    return True


class ReaderInitRequest(BaseModel):
    reading_goal: str = ""


class ReaderProgressRequest(BaseModel):
    chapter: str


@router.post("/{book_id}/init")
def reader_init(book_id: str, req: ReaderInitRequest = Body(default=ReaderInitRequest())) -> dict:
    """Auto-generate a reading plan when user opens a book. Idempotent."""
    book = _resolve(book_id)
    gen = get_plan_editor()
    plan = gen.generate(
        book["title"],
        book_source=book["source"],
        book_id=book_id,
        reading_goal=req.reading_goal,
    )
    get_book_catalog().update_status(book_id, "reading")
    return {"book_id": book_id, "book_title": book["title"], "plan": plan}


@router.get("/{book_id}/plan")
def reader_get_plan(book_id: str) -> dict:
    """Return the current plan markdown for sidebar display."""
    book = _resolve(book_id)
    record = get_plan_catalog().get_by_book_id(book_id)
    content = ""
    if record:
        path = Path(record["file_path"])
        content = path.read_text(encoding="utf-8") if path.exists() else ""
    return {"book_id": book_id, "book_title": book["title"], "plan": content}


@router.post("/{book_id}/progress")
def reader_progress(book_id: str, req: ReaderProgressRequest) -> dict:
    """Mark a chapter as complete, then sync progress to catalog."""
    _resolve(book_id)
    record = get_plan_catalog().get_by_book_id(book_id)
    if not record:
        raise HTTPException(status_code=404, detail="该书暂无阅读计划，请先初始化")

    plan_path = Path(record["file_path"])
    if not _mark_chapter_done(plan_path, req.chapter):
        raise HTTPException(status_code=404, detail=f"未找到章节：{req.chapter}")

    progress = _compute_progress(plan_path)
    get_book_catalog().update_progress(book_id, progress)
    get_plan_catalog().touch(book_id)

    return {
        "ok": True,
        "book_id": book_id,
        "chapter": req.chapter,
        "progress": progress,
    }
