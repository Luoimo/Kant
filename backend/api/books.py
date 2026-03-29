from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from backend.config import get_settings
from backend.rag.chroma.chroma_store import ChromaStore
from backend.rag.extracter.epub_extractor import EpubExtractor
from backend.storage.book_catalog import get_book_catalog

router = APIRouter(prefix="/books", tags=["books"])


class BookEntry(BaseModel):
    id: str
    title: str
    author: str
    source: str
    total_chunks: int
    added_at: str
    cover_path: str
    status: str
    progress: float


class IngestResponse(BaseModel):
    id: str
    source: str
    collection_name: str
    total_chunks: int
    added: int
    skipped: int


@router.get("", response_model=list[BookEntry])
def list_books() -> list[BookEntry]:
    """返回书库中所有书籍的列表。"""
    catalog = get_book_catalog()
    return [
        BookEntry(
            id=b["book_id"],
            title=b["title"],
            author=b["author"],
            source=b["source"],
            total_chunks=b["total_chunks"],
            added_at=b["added_at"],
            cover_path=b["cover_path"],
            status=b["status"],
            progress=b["progress"],
        )
        for b in catalog.get_all()
    ]


@router.post("/upload", response_model=IngestResponse)
async def upload_book(request: Request, file: UploadFile = File(...)) -> IngestResponse:
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

    cover_path = EpubExtractor.extract_cover(dest, settings.covers_dir, result.book_id)

    get_book_catalog().add(
        book_id=result.book_id,
        title=result.book_title,
        author=result.author,
        source=result.source,
        total_chunks=result.total_chunks,
        cover_path=cover_path,
    )

    # 清除 BM25 缓存，确保新书在下次检索时能被纳入索引
    agent = getattr(request.app.state, "agent", None)
    if agent is not None:
        agent.invalidate_retriever()

    return IngestResponse(
        id=result.book_id,
        source=result.source,
        collection_name=result.collection_name,
        total_chunks=result.total_chunks,
        added=result.added,
        skipped=result.skipped,
    )
