# _*_ coding:utf-8 _*_
# 快速验证 RAG 全链路的脚本（支持 Chroma Cloud / 本地双模式）
#
# 功能：
# 1. 扫描 data/books 目录下所有 EPUB
# 2. 对每个 EPUB 执行：提取 → 清洗 → 切块 → 嵌入 → 写入 Chroma
# 3. 用示例查询做检索，打印命中结果
# 4. 测试 Mem0 记忆功能（搜索 / 写入 / 清空）
#
# 运行前确认 .env 已配置：
#   CHROMA_API_KEY / CHROMA_TENANT / CHROMA_DATABASE  （Cloud 模式）
#   或留空 CHROMA_API_KEY 使用本地 PersistentClient   （本地模式）
#   MEM0_API_KEY / MEM0_USER_ID                       （记忆功能，留空则禁用）

import logging
import sys
from pathlib import Path

from backend.config import get_settings
from backend.agents.deepread_agent import DeepReadAgent, DeepReadConfig
from backend.memory.mem0_store import Mem0Store
from backend.rag.chroma import ChromaStore, IngestConfig
from backend.rag.chunker import ChunkConfig
from backend.rag.cleaner import CleanConfig
from backend.rag.retriever import HybridConfig

# ---------------------------------------------------------------------------
# 日志配置：让 backend 各模块的 logging 输出到控制台
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
# 屏蔽 jieba / httpx 等噪音
logging.getLogger("jieba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def build_store() -> ChromaStore:
    """构造 ChromaStore 实例，cloud/本地模式由 .env 自动决定。"""
    settings = get_settings()

    store = ChromaStore(
        chunk_config=ChunkConfig(chunk_size=1000, chunk_overlap=150, section_aware=False),
        clean_config=CleanConfig(),
        ingest_config=IngestConfig(skip_existing=True, embed_batch_size=32),
    )

    mode = "CloudClient" if settings.chroma_api_key else "PersistentClient"
    print(f"[INFO] Chroma 模式：{mode}")
    if settings.chroma_api_key:
        print(f"       tenant   = {settings.chroma_tenant}")
        print(f"       database = {settings.chroma_database}")
    else:
        print(f"       persist  = {store.persist_directory}")

    return store


def ingest_books(store: ChromaStore, books_dir: Path) -> list[str]:
    """将 data/books 下所有 EPUB 写入向量库，返回实际写入的 collection 名称列表。"""
    epub_paths = sorted(books_dir.glob("*.epub"))
    if not epub_paths:
        print(f"[WARN] 目录中没有找到 EPUB：{books_dir}")
        return []

    collection_names = []
    for path in epub_paths:
        print(f"\n=== Ingest: {path.name} ===")
        result = store.ingest(path)
        print(result)
        collection_names.append(result.collection_name)
    return collection_names


def run_query(store: ChromaStore, query: str, collection_name: str, k: int = 5) -> None:
    """纯向量检索并打印结果摘要。"""
    print(f"\n=== Query: {query} ===")
    docs = store.similarity_search(query, k=k, collection_name=collection_name)

    if not docs:
        print("[INFO] 未命中任何结果，请先运行 ingest_books 写入数据。")
        return

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        snippet = doc.page_content.strip().replace("\n", " ")
        print(f"\n--- Result #{i} ---")
        print(f"source     : {meta.get('source')}")
        print(f"chapter    : {meta.get('chapter_title', '')}")
        print(f"section    : {meta.get('section_title', '')}")
        print(f"chunk_index: {meta.get('chunk_index')}")
        print(f"title      : {meta.get('book_title')}")
        print(f"content    : {snippet[:300]}{'...' if len(snippet) > 300 else ''}")


def run_deepread(store: ChromaStore, query: str, collection_name: str,
                 mem0: Mem0Store | None = None) -> None:
    """
    完整混合检索流水线（改写→BM25+向量→RRF→LLM重排→回答）。
    如果传入 mem0，会在调用前搜索历史记忆、调用后保存本次问答。
    """
    print(f"\n{'='*60}")
    print(f"[DeepRead] {query}")
    print("="*60)

    # --- 1. 搜索历史记忆 ---
    memory_context = ""
    if mem0:
        print("\n[Memory] 正在搜索历史记忆...")
        past = mem0.search(query, top_k=3)
        if past:
            print(f"[Memory] 找到 {len(past)} 条相关历史记忆：")
            for i, m in enumerate(past, 1):
                print(f"  [{i}] {m}")
            memory_context = "\n".join(f"- {m}" for m in past)
        else:
            print("[Memory] 暂无相关历史记忆")

    # --- 2. 混合检索 + 回答 ---
    agent = DeepReadAgent(
        store=store,
        collection_name=collection_name,
        config=DeepReadConfig(
            k=5,
            fetch_k=20,
            max_evidence=5,
            consistency_check=False,
            hybrid=HybridConfig(
                fetch_k=20,
                final_k=5,
                enable_query_rewrite=True,
                reranker="llm",
            ),
        ),
    )
    result = agent.run(query=query, memory_context=memory_context)

    print(f"\n[回答]\n{result.answer}")
    if result.citations:
        print("\n[引用来源]")
        for c in result.citations:
            print(f"  - {c}")

    # --- 3. 保存本次问答到记忆 ---
    if mem0 and result.answer:
        print("\n[Memory] 正在保存本次问答...")
        mem0.add_qa(query, result.answer)
        print("[Memory] 已保存 ✓")


def run_memory_status(mem0: Mem0Store, query: str) -> None:
    """仅打印当前与 query 相关的历史记忆，不执行问答，用于调试。"""
    print(f"\n{'='*60}")
    print(f"[Memory Status] 查询：{query}")
    print("="*60)
    past = mem0.search(query, top_k=5)
    if not past:
        print("  (暂无记忆)")
    else:
        for i, m in enumerate(past, 1):
            print(f"  [{i}] {m}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    project_root = Path(__file__).resolve().parent.parent
    books_dir = project_root / "data" / "books"

    store = build_store()

    # 首次运行时取消注释以写入数据；后续查询无需重复写入（skip_existing=True）
    # collection_names = ingest_books(store, books_dir)

    COLLECTION = "book_096b91f931aef93d"  # 替换为你实际的 collection 名

    # --- 初始化 Mem0（API Key 为空时自动禁用，不影响问答流程）---
    mem0 = Mem0Store()

    # --- 查看当前记忆状态（可选，调试用）---
    run_memory_status(mem0, "先验统觉")

    # --- 第一轮：问答 + 自动存入记忆 ---
    # run_deepread(store, "先验统觉是什么？", collection_name=COLLECTION, mem0=mem0)

    # --- 第二轮：同一话题，验证历史记忆是否被检索并注入 ---
    run_deepread(store, "我很喜欢康德，能说说康德的先验统觉是什么吗？", collection_name=COLLECTION, mem0=mem0)

    # --- 清空记忆（谨慎使用）---
    # print("\n[Memory] 清空所有记忆...")
    # mem0.delete_all()
    # print("[Memory] 已清空 ✓")


if __name__ == "__main__":
    main()
