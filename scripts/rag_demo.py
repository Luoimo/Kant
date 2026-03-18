# _*_ coding:utf-8 _*_
# 快速验证 RAG 全链路的脚本（支持 Chroma Cloud / 本地双模式）
#
# 功能：
# 1. 扫描 data/books 目录下所有 PDF
# 2. 对每个 PDF 执行：提取 → 清洗 → 切块 → 嵌入 → 写入 Chroma
# 3. 用示例查询做检索，打印命中结果
#
# 运行前确认 .env 已配置：
#   CHROMA_API_KEY / CHROMA_TENANT / CHROMA_DATABASE  （Cloud 模式）
#   或留空 CHROMA_API_KEY 使用本地 PersistentClient   （本地模式）

import sys
from pathlib import Path

from backend.config import get_settings
from backend.rag.chroma import ChromaStore, IngestConfig
from backend.rag.chunker import ChunkConfig
from backend.rag.cleaner import CleanConfig


def build_store() -> ChromaStore:
    """构造 ChromaStore 实例，cloud/本地模式由 .env 自动决定。"""
    settings = get_settings()

    chunk_cfg = ChunkConfig(
        chunk_size=512,
        chunk_overlap=64,
        page_aware=False,
    )
    ingest_cfg = IngestConfig(
        skip_existing=True,     # 幂等：多次运行不重复写入
        embed_batch_size=32,
    )

    store = ChromaStore(
        collection_name=settings.chroma_database,   # 与 Cloud database 同名，便于对齐
        chunk_config=chunk_cfg,
        clean_config=CleanConfig(),
        ingest_config=ingest_cfg,
    )

    mode = "CloudClient" if settings.chroma_api_key else "PersistentClient"
    print(f"[INFO] Chroma 模式：{mode}")
    if settings.chroma_api_key:
        print(f"       tenant   = {settings.chroma_tenant}")
        print(f"       database = {settings.chroma_database}")
    else:
        print(f"       persist  = {store.persist_directory}")

    return store


def ingest_books(store: ChromaStore, books_dir: Path) -> None:
    """将 data/books 下所有 PDF 写入向量库。"""
    pdf_paths = sorted(books_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"[WARN] 目录中没有找到 PDF：{books_dir}")
        return

    for path in pdf_paths:
        print(f"\n=== Ingest: {path.name} ===")
        result = store.ingest_pdf(path)
        print(result)


def run_query(store: ChromaStore, query: str, k: int = 5) -> None:
    """语义检索并打印结果摘要。"""
    print(f"\n=== Query: {query} ===")
    docs = store.similarity_search(query, k=k)

    if not docs:
        print("[INFO] 未命中任何结果，请先运行 ingest_books 写入数据。")
        return

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        snippet = doc.page_content.strip().replace("\n", " ")
        print(f"\n--- Result #{i} ---")
        print(f"source     : {meta.get('source')}")
        print(f"pages      : {meta.get('page_numbers', '')}")
        print(f"chunk_index: {meta.get('chunk_index')}")
        print(f"title      : {meta.get('pdf_title')}")
        print(f"author     : {meta.get('pdf_author')}")
        print(f"content    : {snippet[:300]}{'...' if len(snippet) > 300 else ''}")


def main() -> None:
    # Windows 控制台默认编码可能导致中文/特殊符号输出失败，这里尽量统一为 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    project_root = Path(__file__).resolve().parent.parent
    books_dir = project_root / "data" / "books"

    store = build_store()

    # 首次运行时取消注释以写入数据；后续查询无需重复写入（skip_existing=True）
    ingest_books(store, books_dir)

    run_query(store, "克尔凯郭尔对焦虑的定义是什么？", k=5)
    run_query(store, "克尔凯郭尔的父亲是个怎样的人？", k=5)


if __name__ == "__main__":
    main()
