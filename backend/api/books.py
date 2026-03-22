from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.agents.orchestrator_agent import invalidate_bm25_caches
from backend.config import get_settings
from backend.rag.chroma.chroma_store import ChromaStore

router = APIRouter(prefix="/books", tags=["books"])


class BookEntry(BaseModel):
    id: str           # book_id (UUID)
    source: str       # ChromaDB source 路径（调试用）
    book_title: str
    author: str


class IngestResponse(BaseModel):
    id: str           # book_id (UUID)
    source: str
    collection_name: str
    total_chunks: int
    added: int
    skipped: int


@router.get("", response_model=list[BookEntry])
def list_books() -> list[BookEntry]:
    """返回书库中所有书籍的列表。"""
    store = ChromaStore()
    return [
        BookEntry(
            id=entry.get("book_id", ""),
            source=entry.get("source", ""),
            book_title=entry.get("book_title", ""),
            author=entry.get("author", ""),
        )
        for entry in store.list_book_titles()
    ]


@router.post("/upload", response_model=IngestResponse)
async def upload_book(file: UploadFile = File(...)) -> IngestResponse:
    """上传 EPUB 文件并触发入库流水线。"""
    filename = file.filename or ""
    if not filename.lower().endswith(".epub"):
        raise HTTPException(status_code=400, detail="仅支持 .epub 格式文件")

    settings = get_settings()
    books_dir = Path(settings.books_data_dir)
    books_dir.mkdir(parents=True, exist_ok=True)
    dest = books_dir / filename

    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        shutil.move(str(tmp_path), dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="文件保存失败")

    try:
        store = ChromaStore()
        result = store.ingest(dest)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"入库失败：{exc}") from exc

    invalidate_bm25_caches()

    return IngestResponse(
        id=result.book_id,
        source=result.source,
        collection_name=result.collection_name,
        total_chunks=result.total_chunks,
        added=result.added,
        skipped=result.skipped,
    )
