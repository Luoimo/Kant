# _*_ coding:utf-8 _*_
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from langchain_core.documents import Document

from .bm25_retriever import BM25Retriever
from .query_rewriter import QueryRewriter
from .reranker import CrossEncoderReranker, LLMReranker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class HybridConfig:
    """
    混合检索流水线配置。

    fetch_k             : 向量检索和 BM25 各自召回的候选数量
    final_k             : rerank 后最终保留的文档数量
    rrf_k               : RRF 公式中的平滑常数（默认 60）
    enable_query_rewrite: 是否启用 LLM 查询改写
    reranker            : "llm"（默认）| "cross_encoder" | "none"
    cross_encoder_model : cross_encoder 模式下使用的模型名
    """
    fetch_k: int = 20
    final_k: int = 6
    rrf_k: int = 60
    enable_query_rewrite: bool = True
    reranker: str = "llm"
    cross_encoder_model: str = "BAAI/bge-reranker-base"


# ---------------------------------------------------------------------------
# RRF 融合
# ---------------------------------------------------------------------------

def _rrf_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    rrf_k: int = 60,
) -> dict[str, float]:
    """
    Reciprocal Rank Fusion：将多个排序列表合并为一个综合分数字典。

    score(d) = Σ  1 / (k + rank(d))
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return scores


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    混合检索器：向量检索 + BM25，通过 RRF 融合，再经 LLM / CrossEncoder 重排。

    BM25 索引在首次 ``search`` 调用时懒加载（从 ChromaDB 拉取全量文档构建），
    后续查询复用同一索引，无需重建。

    示例::

        retriever = HybridRetriever(
            store=chroma_store,
            collection_name="book_f24c24e0ec71320d",
            config=HybridConfig(fetch_k=20, final_k=6, reranker="llm"),
        )
        docs = retriever.search("先验统觉是什么？")
    """

    def __init__(
        self,
        store,                          # ChromaStore，避免循环引用不直接 type hint
        collection_name: str,
        config: HybridConfig | None = None,
        llm=None,
    ) -> None:
        self._store = store
        self._collection_name = collection_name
        self.config = config or HybridConfig()

        self._query_rewriter = (
            QueryRewriter(llm=llm) if self.config.enable_query_rewrite else None
        )
        self._reranker = self._build_reranker(llm)
        self._bm25_cache: dict[str, BM25Retriever] = {}  # filter_key → BM25Retriever
        self._executor = ThreadPoolExecutor(max_workers=2)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        filter: dict | None = None,
    ) -> list[Document]:
        """
        完整检索流水线：改写 → 混合检索 → RRF → rerank → TopK。

        :param query:  用户原始问题
        :param filter: Chroma where 过滤（例如 ``{"source": "path/to/book.epub"}``）
        """
        cfg = self.config

        # 1) Query Rewrite
        rewritten = self._query_rewriter.rewrite(query) if self._query_rewriter else query
        if rewritten != query:
            logger.info("QueryRewrite: %r → %r", query, rewritten)

        # 2+3) 向量检索与 BM25 检索并行执行
        bm25 = self._get_bm25(filter)
        _f_vec = self._executor.submit(
            self._store.similarity_search_with_score,
            rewritten, k=cfg.fetch_k, filter=filter,
            collection_name=self._collection_name,
        )
        _f_bm25 = self._executor.submit(bm25.search, rewritten, k=cfg.fetch_k)
        vector_results = _f_vec.result()
        bm25_results = _f_bm25.result()

        # 4) RRF 融合
        all_docs = _dedupe(vector_results + bm25_results)
        vec_ranked = _to_id_score_list(vector_results)
        bm25_ranked = _to_id_score_list(bm25_results)
        rrf_scores = _rrf_fusion([vec_ranked, bm25_ranked], rrf_k=cfg.rrf_k)

        # 按 RRF 分数排序，取候选池（供 reranker 使用）
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        candidates = [all_docs[cid] for cid in sorted_ids if cid in all_docs]

        # 5) Rerank
        if self._reranker is not None:
            final_docs = self._reranker.rerank(query, candidates, top_k=cfg.final_k)
        else:
            final_docs = candidates[: cfg.final_k]

        logger.info(
            "HybridRetriever: vector=%d, bm25=%d, rrf=%d → rerank → final=%d",
            len(vector_results),
            len(bm25_results),
            len(candidates),
            len(final_docs),
        )
        return final_docs

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def invalidate_bm25(self) -> None:
        """入库新书后调用，清除所有 BM25 缓存，下次查询时重建。"""
        self._bm25_cache.clear()

    def _get_bm25(self, filter: dict | None) -> BM25Retriever:
        """按 filter 分开缓存 BM25 索引，不同书源互不污染。"""
        cache_key = json.dumps(filter, sort_keys=True) if filter else "__all__"
        if cache_key not in self._bm25_cache:
            logger.info("构建 BM25 索引（collection=%s, filter=%s）...", self._collection_name, cache_key)
            docs = self._store.get_all_documents(
                collection_name=self._collection_name,
                filter=filter,
            )
            self._bm25_cache[cache_key] = BM25Retriever(docs)
            logger.info("BM25 索引就绪，共 %d 个 chunk（key=%s）", len(docs), cache_key)
        return self._bm25_cache[cache_key]

    def _build_reranker(self, llm):
        mode = self.config.reranker
        if mode == "cross_encoder":
            return CrossEncoderReranker(self.config.cross_encoder_model)
        if mode == "llm":
            return LLMReranker(llm=llm)
        return None  # "none"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _doc_id(doc: Document) -> str:
    return doc.metadata.get("chunk_id") or doc.page_content[:40]


def _dedupe(
    results: list[tuple[Document, float]],
) -> dict[str, Document]:
    seen: dict[str, Document] = {}
    for doc, _ in results:
        did = _doc_id(doc)
        if did not in seen:
            seen[did] = doc
    return seen


def _to_id_score_list(
    results: list[tuple[Document, float]],
) -> list[tuple[str, float]]:
    return [(_doc_id(doc), score) for doc, score in results]
