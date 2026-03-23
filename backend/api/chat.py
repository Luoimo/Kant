from __future__ import annotations

import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.agents.orchestrator_agent import invalidate_bm25_caches, run_minimal_graph
from backend.config import get_settings
from backend.rag.chroma.chroma_store import ChromaStore
from backend.xai.citation import Citation

app = FastAPI(title="Kant Reading Agent")


class ChatRequest(BaseModel):
    query: str
    book_source: str | None = None
    thread_id: str = "default"
    # Reader 模式专用：前端传入当前 tab / 划选原文 / 当前章节
    active_tab: str | None = None
    selected_text: str | None = None
    current_chapter: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    retrieved_docs_count: int
    intent: str | None


def _citation_to_dict(c: Any) -> dict[str, Any]:
    if isinstance(c, Citation):
        return asdict(c)
    if isinstance(c, dict):
        return c
    return {"value": str(c)}


class IngestResponse(BaseModel):
    source: str
    collection_name: str
    total_chunks: int
    added: int
    skipped: int


@app.post("/books/upload", response_model=IngestResponse)
async def upload_book(file: UploadFile = File(...)) -> IngestResponse:
    """
    上传 EPUB 文件并触发入库流水线。

    流程：
    1. 将文件持久化保存到 BOOKS_DATA_DIR
    2. 运行 ChromaStore.ingest()（切块 → 向量化 → 写入主集合）
    3. 同步更新 book_catalog 集合（upsert 书目摘要条目）

    返回 IngestResult 统计信息及 catalog 更新状态。
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".epub"):
        raise HTTPException(status_code=400, detail="仅支持 .epub 格式文件")

    settings = get_settings()
    books_dir = Path(settings.books_data_dir)
    books_dir.mkdir(parents=True, exist_ok=True)
    dest = books_dir / filename

    # 先存临时文件，成功后再移动（避免写到一半的残缺文件留在 books_dir）
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

    # 入库成功后清除 BM25 缓存，确保下次检索包含新书内容
    invalidate_bm25_caches()

    return IngestResponse(
        source=result.source,
        collection_name=result.collection_name,
        total_chunks=result.total_chunks,
        added=result.added,
        skipped=result.skipped,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    state = run_minimal_graph(
        req.query,
        book_source=req.book_source,
        thread_id=req.thread_id,
        active_tab=req.active_tab,
        selected_text=req.selected_text,
        current_chapter=req.current_chapter,
    )
    return ChatResponse(
        answer=state.get("answer", ""),
        citations=[_citation_to_dict(c) for c in state.get("citations", [])],
        retrieved_docs_count=state.get("retrieved_docs_count", 0),
        intent=state.get("intent"),
    )
