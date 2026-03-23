# _*_ coding:utf-8 _*_
# @Time:2026/3/10
# @Author:Chloe
# @File:chroma_store.py
# @Project:Kant

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document

from backend.config import get_settings
from backend.llm.openai_client import get_embeddings
from backend.rag.chunker.text_chunker import ChunkConfig, TextChunk, TextChunker
from backend.rag.cleaner.text_cleaner import CleanConfig, TextCleaner
from backend.rag.extracter.epub_extractor import EpubExtractor

logger = logging.getLogger(__name__)

# Chroma 元数据值只允许 str/int/float/bool，list 需序列化
_PAGE_SEP = ","


def _sanitize_collection_name(title: str) -> str:
    """
    将书名转为合法的 Chroma collection 名称。

    统一使用 ``book_{md5前16位}`` 格式，确保：
    - 无论中英文书名都合法
    - 同一书名始终映射到同一 collection（稳定唯一）
    """
    hash16 = hashlib.md5(title.encode("utf-8")).hexdigest()[:16]
    return f"book_{hash16}"


# ---------------------------------------------------------------------------
# chromadb 原生包装器（替代 langchain_chroma.Chroma）
# ---------------------------------------------------------------------------

class Chroma:
    """
    基于 chromadb 原生 SDK 的向量存储，提供与 langchain_chroma.Chroma 兼容的公开接口。

    客户端选择策略：
    * ``api_key`` 非空 → ``chromadb.CloudClient``（Chroma Cloud 托管，读取 tenant/database）
    * ``api_key`` 为空 → ``chromadb.PersistentClient``（本地持久化，读取 persist_directory）

    公开接口：

    * ``add_documents``                —— 批量嵌入并写入 collection
    * ``similarity_search``            —— 语义检索，返回 Document 列表
    * ``similarity_search_with_score`` —— 带余弦距离分数的语义检索
    * ``as_retriever``                 —— 返回 LangChain 兼容的 Retriever
    * ``_collection``                  —— 底层 chromadb.Collection，供内部 CRUD 直接使用
    """

    def __init__(
        self,
        collection_name: str,
        embedding_function,
        persist_directory: str,
        api_key: str = "",
        tenant: str = "default_tenant",
        database: str = "default_database",
    ) -> None:
        self._embedding_function = embedding_function

        if api_key:
            logger.debug(
                "Chroma 使用 CloudClient（tenant=%s, database=%s）", tenant, database
            )
            self._client = chromadb.CloudClient(
                tenant=tenant,
                database=database,
                api_key=api_key,
            )
        else:
            logger.debug("Chroma 使用 PersistentClient（path=%s）", persist_directory)
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_directory)

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # --- 写入 ---

    def upsert_documents(self, documents: list[Document], ids: list[str]) -> None:
        """向量化并 upsert（存在则更新，不存在则插入），用于 book_catalog 条目维护。"""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = self._embedding_function.embed_documents(texts)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        """嵌入文本并批量写入 chromadb collection。"""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = self._embedding_function.embed_documents(texts)
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    # --- 检索 ---

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
    ) -> list[Document]:
        """语义相似度检索，返回最相关的 k 个 Document。"""
        results = self._query(query, k=k, where=filter)
        return [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
    ) -> list[tuple[Document, float]]:
        """带余弦距离分数的语义检索（分数越低越相似）。"""
        results = self._query(query, k=k, where=filter)
        return [
            (Document(page_content=doc, metadata=meta), dist)
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def as_retriever(self, **kwargs: Any):
        """返回 LangChain 兼容的 BaseRetriever，可直接接入 Chain / Agent。"""
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

        search_kwargs: dict = kwargs.get("search_kwargs", {})
        k: int = search_kwargs.get("k", 4)
        flt: dict | None = search_kwargs.get("filter")
        store = self

        class _Retriever(BaseRetriever):
            def _get_relevant_documents(
                self_,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun,
            ) -> list[Document]:
                return store.similarity_search(query, k=k, filter=flt)

        return _Retriever()

    # --- 内部 ---

    def _query(self, query: str, k: int, where: dict | None) -> dict:
        """调用 chromadb 向量查询，返回原始结果字典。"""
        embedding = self._embedding_function.embed_query(query)
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where
        return self._collection.query(**query_kwargs)


# ---------------------------------------------------------------------------
# 辅助数据类
# ---------------------------------------------------------------------------

@dataclass
class IngestConfig:
    """
    ingest / ingest_chunks 的行为配置。

    skip_existing   : True → collection 已有数据时整本跳过（幂等写入）
    embed_batch_size: 每批次送入 Embedding API 及 Chroma 的 chunk 数量，避免超出请求限制
    """
    skip_existing: bool = True
    embed_batch_size: int = 100


@dataclass
class IngestResult:
    """一次 ingest 操作的统计摘要。"""
    source: str
    total_chunks: int       # 切分后总 chunk 数
    added: int              # 本次实际写入数量
    skipped: int            # 因去重跳过的数量
    collection_name: str

    def __str__(self) -> str:
        return (
            f"[{self.collection_name}] {Path(self.source).name} → "
            f"总计 {self.total_chunks} 块，写入 {self.added}，跳过 {self.skipped}"
        )


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class ChromaStore:
    """
    EPUB 向量化存储管理类，封装「提取 → 清洗 → 切块 → 向量化 → 存储」全流水线。

    使用 :class:`Chroma`（基于 chromadb 原生 SDK）作为向量库后端，
    使用 :func:`~backend.llm.openai_client.get_embeddings` 提供的 OpenAI Embedding
    模型将 chunk 向量化，并持久化到 ``chroma_persist_dir``（来自 config）。

    主要功能：

    * ``ingest``        ——  一键完成 EPUB → 向量库全流程
    * ``ingest_chunks`` ——  直接写入已处理好的 :class:`~backend.rag.chunker.text_chunker.TextChunk`
    * ``delete_source`` ——  按来源文件路径删除所有关联 chunk
    * ``similarity_search`` / ``as_retriever`` ——  语义检索接口

    示例::

        store = ChromaStore(collection_name="kant")

        # 一键入库
        result = store.ingest("data/books/kant.epub")
        print(result)   # [kant] kant.epub → 总计 312 块，写入 312，跳过 0

        # 语义检索
        docs = store.similarity_search("什么是纯粹理性？", k=5)

        # 作为 LangChain Retriever 使用
        retriever = store.as_retriever(search_kwargs={"k": 5})
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: str | None = None,
        chunk_config: ChunkConfig | None = None,
        clean_config: CleanConfig | None = None,
        ingest_config: IngestConfig | None = None,
    ) -> None:
        settings = get_settings()
        self.collection_name = collection_name or settings.books_collection_name
        self.persist_directory = persist_directory or str(
            Path(settings.chroma_persist_dir).resolve()
        )
        self.chunk_config = chunk_config or ChunkConfig()
        self.clean_config = clean_config or CleanConfig()
        self.ingest_config = ingest_config or IngestConfig()

        # Chroma Cloud 凭据（来自 .env）
        self._chroma_api_key = settings.chroma_api_key
        self._chroma_tenant = settings.chroma_tenant
        self._chroma_database = settings.chroma_database

        self._embeddings = get_embeddings()
        self._db = self._get_or_create_db()

    # ------------------------------------------------------------------
    # 公开：写入接口
    # ------------------------------------------------------------------

    def ingest(
        self,
        path: str | Path,
        *,
        collection_name: str | None = None,
    ) -> IngestResult:
        """
        EPUB 全流水线入库：提取 → 清洗 → 切块 → 向量化 → 写入 Chroma。

        :param path:            EPUB 文件路径
        :param collection_name: 指定 collection 名称；None 则自动取书名（推荐）
        """
        path = Path(path)
        logger.info("开始入库流水线：%s", path.name)

        extractor = EpubExtractor(path)
        book_content = extractor.extract()
        logger.debug("  ✓ 提取完成，共 %d 章节", len(book_content.sections))

        cleaner = TextCleaner(self.clean_config)
        cleaned = cleaner.clean_content(book_content)
        logger.debug("  ✓ 清洗完成")

        chunker = TextChunker(self.chunk_config)
        chunks = chunker.chunk_content(cleaned)
        logger.debug("  ✓ 切块完成，共 %d 个 chunk", len(chunks))

        # 未指定 collection 时，使用 ChromaStore 默认 collection（所有书共享一个库）
        if collection_name is None:
            collection_name = self.collection_name
        logger.info("  → collection：%s", collection_name)

        db = self._resolve_db(collection_name)
        result = self._ingest_chunks_to_db(chunks, db, source=str(path))
        logger.info("  ✓ 入库完成：%s", result)

        return result

    def ingest_chunks(
        self,
        chunks: list[TextChunk],
        *,
        collection_name: str | None = None,
    ) -> IngestResult:
        """
        直接写入预处理好的 :class:`TextChunk` 列表（跳过提取/清洗/切块步骤）。
        适合外部已完成处理、或对流水线有定制需求的场景。
        """
        db = self._resolve_db(collection_name)
        source = chunks[0].metadata.source if chunks else ""
        return self._ingest_chunks_to_db(chunks, db, source=source)

    # ------------------------------------------------------------------
    # 公开：删除接口
    # ------------------------------------------------------------------

    def delete_source(self, source: str) -> int:
        """
        删除指定来源文件的所有 chunk。

        :param source: PDF 文件路径字符串（与入库时 metadata.source 完全匹配）
        :returns:      实际删除的文档数量
        """
        collection = self._db._collection
        results = collection.get(where={"source": source})
        ids = results.get("ids", [])
        if ids:
            collection.delete(ids=ids)
            logger.info("已删除 %d 个 chunk（source=%s）", len(ids), source)
        return len(ids)

    # ------------------------------------------------------------------
    # 公开：检索接口
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        collection_name: str | None = None,
    ) -> list[Document]:
        """
        语义相似度检索，返回 LangChain :class:`~langchain_core.documents.Document` 列表。

        :param query:           查询文本
        :param k:               返回的最相关 chunk 数量
        :param filter:          Chroma where 过滤条件，例如 ``{"book_title": "Critique of Pure Reason"}``
        :param collection_name: 指定查询的 collection；None 则使用默认 collection
        """
        db = self._resolve_db(collection_name)
        return db.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        collection_name: str | None = None,
    ) -> list[tuple[Document, float]]:
        """带相似度分数的检索，分数越低（余弦距离）表示越相关。"""
        db = self._resolve_db(collection_name)
        return db.similarity_search_with_score(query, k=k, filter=filter)

    def as_retriever(self, **kwargs: Any):
        """
        返回 LangChain Retriever 对象，可直接接入 Chain / Agent。

        常用参数::

            retriever = store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5},
            )
        """
        return self._db.as_retriever(**kwargs)

    # ------------------------------------------------------------------
    # 公开：信息查询
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """返回当前 collection 的基本统计信息。"""
        collection = self._db._collection
        count = collection.count()
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "total_chunks": count,
        }

    def get_all_documents(
        self,
        collection_name: str | None = None,
        filter: dict | None = None,
    ) -> list[Document]:
        """
        返回 collection 中的所有文档（用于构建 BM25 索引等离线任务）。

        :param collection_name: 指定 collection；None 则使用默认
        :param filter:          Chroma where 条件，例如 ``{"source": "path/to/book.epub"}``
        """
        db = self._resolve_db(collection_name)
        get_kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        if filter:
            get_kwargs["where"] = filter
        results = db._collection.get(**get_kwargs)
        return [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(
                results.get("documents") or [],
                results.get("metadatas") or [],
            )
        ]

    def list_sources(self) -> list[str]:
        """列出当前 collection 中所有已入库的文件来源路径（去重）。"""
        collection = self._db._collection
        results = collection.get(include=["metadatas"])
        sources = {
            meta.get("source", "")
            for meta in (results.get("metadatas") or [])
            if meta
        }
        return sorted(sources - {""})

    def list_book_titles(self) -> list[dict[str, str]]:
        """
        返回库中所有书的 {book_title, author, source} 列表（去重）。

        每本书从主 collection 中取一条 metadata 即可，不需要向量查询。
        用于 RecommendationAgent 标注推荐结果是否已在库中。
        """
        sources = self.list_sources()
        books: list[dict[str, str]] = []
        seen: set[str] = set()
        for src in sources:
            try:
                results = self._db._collection.get(
                    where={"source": src},
                    include=["metadatas"],
                    limit=1,
                )
                metas = results.get("metadatas") or []
                if metas:
                    title = metas[0].get("book_title", "")
                    author = metas[0].get("author", "")
                    if title and title not in seen:
                        seen.add(title)
                        books.append({"book_title": title, "author": author, "source": src})
            except Exception as exc:
                logger.warning("list_book_titles：跳过 source=%s，原因：%s", src, exc)
        return books

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _get_or_create_db(self) -> Chroma:
        """初始化或加载已有的 Chroma 实例（Cloud 或本地）。"""
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=self.persist_directory,
            api_key=self._chroma_api_key,
            tenant=self._chroma_tenant,
            database=self._chroma_database,
        )

    def _resolve_db(self, collection_name: str | None) -> Chroma:
        """如果指定了临时 collection_name，创建对应的 Chroma 实例；否则返回默认实例。"""
        if collection_name and collection_name != self.collection_name:
            return Chroma(
                collection_name=collection_name,
                embedding_function=self._embeddings,
                persist_directory=self.persist_directory,
                api_key=self._chroma_api_key,
                tenant=self._chroma_tenant,
                database=self._chroma_database,
            )
        return self._db

    def _ingest_chunks_to_db(
        self,
        chunks: list[TextChunk],
        db: Chroma,
        source: str,
    ) -> IngestResult:
        """
        将 TextChunk 列表写入 Chroma，支持跳过已有 collection 和分批 Embedding。
        """
        cfg = self.ingest_config
        total = len(chunks)

        # 去重：按 chunk_id 逐条比对，只写入新增 chunk（支持书籍增量更新）
        # 分批查询以避免 Chroma Cloud 单次 Get 请求的记录数限制（免费套餐 ≤300 条/次）
        if cfg.skip_existing and total > 0:
            all_ids = [c.chunk_id for c in chunks]
            existing_ids: set[str] = set()
            for i in range(0, len(all_ids), cfg.embed_batch_size):
                batch_ids = all_ids[i: i + cfg.embed_batch_size]
                result = db._collection.get(ids=batch_ids, include=[])
                existing_ids.update(result.get("ids") or [])
            chunks = [c for c in chunks if c.chunk_id not in existing_ids]
            skipped = total - len(chunks)
            if skipped:
                logger.info("  → 跳过已有 chunk：%d 条", skipped)
        else:
            skipped = 0

        if not chunks:
            return IngestResult(
                source=source,
                total_chunks=total,
                added=0,
                skipped=total,
                collection_name=db._collection.name,
            )

        # 分批写入
        added = 0
        for i in range(0, len(chunks), cfg.embed_batch_size):
            batch = chunks[i: i + cfg.embed_batch_size]
            documents = [self._chunk_to_document(c) for c in batch]
            ids = [c.chunk_id for c in batch]
            db.add_documents(documents=documents, ids=ids)
            added += len(batch)
            logger.debug("  写入进度：%d / %d", added, total)

        return IngestResult(
            source=source,
            total_chunks=total,
            added=added,
            skipped=skipped,
            collection_name=db._collection.name,
        )

    @staticmethod
    def _chunk_to_document(chunk: TextChunk) -> Document:
        """
        将 :class:`TextChunk` 转换为 LangChain :class:`Document`。
        Chroma 元数据值必须为 str/int/float/bool，page_numbers 序列化为逗号分隔字符串。
        """
        meta = chunk.metadata
        return Document(
            page_content=chunk.text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "char_count": chunk.char_count,
                "source": meta.source,
                "section_indices": _PAGE_SEP.join(str(i) for i in meta.section_indices),
                "chunk_index": meta.chunk_index,
                "book_title": meta.book_title,
                "author": meta.author,
                "chapter_title": getattr(meta, "chapter_title", "") or "",
                "section_title": getattr(meta, "section_title", "") or "",
            },
        )
