from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Request
from pydantic import BaseModel

from config import get_settings
from graph.neo4j_store import get_neo4j_store
from rag.chroma.chroma_store import ChromaStore
from rag.extracter.epub_extractor import EpubExtractor
from api.deps import require_member
from storage.book_catalog import get_book_catalog, get_note_catalog
from storage.oss_client import (
    build_oss_uri,
    get_oss_client,
    is_oss_uri,
    parse_oss_uri,
)

logger = logging.getLogger(__name__)

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


class DeleteResponse(BaseModel):
    id: str
    deleted_chunks: int
    removed_source_file: bool
    removed_cover: bool
    removed_note: bool


# ---------------------------------------------------------------------------
# 工具：把 catalog 中存储的 cover_path/source 转换成前端可用的访问 URL
# ---------------------------------------------------------------------------

def _resolve_public_url(value: str) -> str:
    """
    catalog 中的 ``cover_path`` / ``source`` 可能是：
    - ``oss://bucket/key`` → 生成签名 URL 供前端直接读取
    - 本地路径（兼容旧数据）→ 原样返回，交给前端的 StaticFiles 逻辑处理
    - 空字符串 → 原样返回
    """
    if not value:
        return ""
    if is_oss_uri(value):
        try:
            return get_oss_client().signed_url_from_uri(value)
        except Exception:
            logger.exception("生成 OSS 签名 URL 失败：%s", value)
            return value
    return value


@router.get("", response_model=list[BookEntry])
def list_books(current_user: dict = Depends(require_member)) -> list[BookEntry]:
    """返回书库中所有书籍的列表。封面与源文件字段会被转换为签名 URL。"""
    catalog = get_book_catalog()
    owner = current_user["user_id"]
    return [
        BookEntry(
            id=b["book_id"],
            title=b["title"],
            author=b["author"],
            source=_resolve_public_url(b["source"]),
            total_chunks=b["total_chunks"],
            added_at=b["added_at"],
            cover_path=_resolve_public_url(b["cover_path"]),
            status=b["status"],
            progress=b["progress"],
        )
        for b in catalog.get_all(owner_user_id=owner)
    ]


@router.get("/{book_id}/file")
def get_book_file(book_id: str, current_user: dict = Depends(require_member)) -> dict:
    """返回 EPUB 文件的临时签名 URL，供前端阅读器直接加载。

    对兼容旧数据（source 为本地路径）的场景，直接返回原 source。
    """
    catalog = get_book_catalog()
    book = catalog.get_by_id(book_id, owner_user_id=current_user["user_id"])
    if not book:
        raise HTTPException(status_code=404, detail="书籍不存在")
    source = book.get("source") or ""
    return {"id": book_id, "url": _resolve_public_url(source)}


@router.post("/upload", response_model=IngestResponse)
async def upload_book(
    file: UploadFile = File(...),
    current_user: dict = Depends(require_member),
) -> IngestResponse:
    """上传 EPUB 文件并触发入库流水线。

    流程：
    1. 把上传文件落到临时文件（EpubExtractor 和 ChromaStore 需要本地路径）
    2. 上传到 OSS：``oss://{bucket}/{books_prefix}/{filename}``
    3. 用 OSS URI 作为 source_override 调用 ChromaStore.ingest
    4. 提取封面 → 上传到 OSS → 拿到 ``oss://{bucket}/{covers_prefix}/{book_id}.xxx``
    5. 写入 book_catalog、Neo4j
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".epub"):
        raise HTTPException(status_code=400, detail="仅支持 .epub 格式文件")

    settings = get_settings()
    oss = get_oss_client()
    owner = current_user["user_id"]

    # 1. 先把上传流落到临时文件
    tmp_dir = Path(tempfile.mkdtemp(prefix="kant_upload_"))
    tmp_epub = tmp_dir / filename
    try:
        with tmp_epub.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"文件保存失败：{exc}") from exc

    try:
        # 2. 计算 source（OSS 优先，未启用时退回本地路径）
        if oss.enabled:
            book_key = oss.book_key(owner, filename)
            source_uri = build_oss_uri(oss.bucket_name, book_key)
        else:
            local_books_dir = Path(settings.books_data_dir)
            local_books_dir.mkdir(parents=True, exist_ok=True)
            local_dest = local_books_dir / filename
            shutil.copy2(tmp_epub, local_dest)
            source_uri = str(local_dest)

        # 3. 入库到 Chroma（基于临时文件解析，但 source 写成 OSS URI）
        try:
            store = ChromaStore()
            result = store.ingest(tmp_epub, source_override=source_uri, owner_user_id=owner)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"入库失败：{exc}") from exc

        # 4. 上传 EPUB 原文件到 OSS（入库成功后再上传，失败时不污染 OSS）
        if oss.enabled:
            oss.put_file(book_key, tmp_epub, content_type="application/epub+zip")

        # 5. 提取封面
        cover_value = ""
        if oss.enabled:
            # 封面先落临时目录，再上传到 OSS
            cover_tmp = EpubExtractor.extract_cover(
                tmp_epub, tmp_dir, result.book_id
            )
            if cover_tmp:
                cover_tmp_path = Path(cover_tmp)
                cover_key = oss.cover_key(owner, result.book_id, cover_tmp_path.suffix)
                content_type = _guess_image_content_type(cover_tmp_path.suffix)
                oss.put_file(cover_key, cover_tmp_path, content_type=content_type)
                cover_value = build_oss_uri(oss.bucket_name, cover_key)
        else:
            cover_value = EpubExtractor.extract_cover(
                tmp_epub, settings.covers_dir, result.book_id
            )

        # 6. 写入 catalog / Neo4j（source 与 cover 都用 OSS URI）
        get_book_catalog().add(
            owner_user_id=owner,
            book_id=result.book_id,
            title=result.book_title,
            author=result.author,
            source=source_uri,
            total_chunks=result.total_chunks,
            cover_path=cover_value,
        )
        get_neo4j_store().upsert_book(
            book_id=result.book_id,
            title=result.book_title,
            author=result.author,
            source=source_uri,
            total_chunks=result.total_chunks,
            cover_path=cover_value,
        )
        graph_docs = store.get_all_documents(filter={"book_id": result.book_id, "owner_user_id": owner})
        get_neo4j_store().upsert_book_graph(
            book_id=result.book_id,
            documents=graph_docs,
        )

        return IngestResponse(
            id=result.book_id,
            source=_resolve_public_url(source_uri),
            collection_name=result.collection_name,
            total_chunks=result.total_chunks,
            added=result.added,
            skipped=result.skipped,
        )
    finally:
        # 7. 清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.delete("/{book_id}", response_model=DeleteResponse)
def delete_book(
    book_id: str,
    request: Request,
    current_user: dict = Depends(require_member),
) -> DeleteResponse:
    """级联删除指定书籍：向量库 chunk、图谱、目录、封面、源文件、笔记、聊天记录。"""
    catalog = get_book_catalog()
    owner = current_user["user_id"]
    book = catalog.get_by_id(book_id, owner_user_id=owner)
    if not book:
        raise HTTPException(status_code=404, detail="书籍不存在")

    # 清理聊天记录
    agent = getattr(request.app.state, "agent", None)
    if agent and hasattr(agent, "clear_chat_history"):
        # 会话化后需要按 conversation_id 清理，此处不做全删，保留由会话接口专门处理。
        pass

    source = book.get("source") or ""
    cover_path = book.get("cover_path") or ""

    deleted_chunks = 0
    try:
        store = ChromaStore()
        if source:
            deleted_chunks = store.delete_source(source, owner_user_id=owner)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"向量库清理失败：{exc}") from exc

    try:
        get_neo4j_store().delete_book(book_id=book_id)
    except Exception:
        # Neo4j 未连接或异常已在内部降级，这里再兜底一次。
        pass

    note_catalog = get_note_catalog()
    note_meta = note_catalog.get_by_book_id(book_id, owner_user_id=owner)
    removed_note = False
    if note_meta:
        note_file = Path(note_meta.get("file_path") or "")
        if note_file.is_file():
            try:
                note_file.unlink()
                removed_note = True
            except OSError:
                removed_note = False
        note_catalog.delete(book_id, owner_user_id=owner)

    # 删除 EPUB 源文件：区分 OSS URI 与本地路径
    removed_source_file = _remove_asset(source)

    # 删除封面：同样区分 OSS URI 与本地路径
    removed_cover = _remove_asset(
        cover_path,
        local_fallback_dir=get_settings().covers_dir,
    )

    catalog.delete(book_id, owner_user_id=owner)

    return DeleteResponse(
        id=book_id,
        deleted_chunks=deleted_chunks,
        removed_source_file=removed_source_file,
        removed_cover=removed_cover,
        removed_note=removed_note,
    )


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _remove_asset(path_or_uri: str, *, local_fallback_dir: str | None = None) -> bool:
    """统一删除 ``oss://...`` / 本地文件；找不到返回 False。"""
    if not path_or_uri:
        return False

    if is_oss_uri(path_or_uri):
        try:
            _, key = parse_oss_uri(path_or_uri)
            return get_oss_client().delete(key)
        except Exception:
            logger.exception("删除 OSS 对象失败：%s", path_or_uri)
            return False

    local = Path(path_or_uri)
    if not local.is_absolute() and local_fallback_dir:
        local = Path(local_fallback_dir) / local.name
    if local.is_file():
        try:
            local.unlink()
            return True
        except OSError:
            return False
    return False


def _guess_image_content_type(suffix: str) -> str:
    s = suffix.lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(s, "application/octet-stream")
