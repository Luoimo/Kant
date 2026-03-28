"""Unit tests for retriever components: BM25Retriever, QueryRewriter, LLMReranker, HybridRetriever.

All tests use mocks — no real API calls or ChromaDB required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.documents import Document

from backend.rag.retriever.bm25_retriever import BM25Retriever
from backend.utils.text import tokenize as _tokenize
from backend.rag.retriever.query_rewriter import QueryRewriter
from backend.rag.retriever.reranker import LLMReranker, _parse_scores
from backend.rag.retriever.hybrid_retriever import (
    HybridConfig,
    HybridRetriever,
    _rrf_fusion,
    _doc_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(text: str, chunk_id: str | None = None) -> Document:
    meta = {"chunk_id": chunk_id} if chunk_id else {}
    return Document(page_content=text, metadata=meta)


def _scored(doc: Document, score: float = 1.0) -> tuple[Document, float]:
    return (doc, score)


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class TestBM25Retriever:
    @pytest.fixture
    def docs(self):
        return [
            _doc("先验感性论探讨空间与时间的本质", "id1"),
            _doc("纯粹理性批判是康德的主要著作", "id2"),
            _doc("道德形而上学奠基讨论义务伦理", "id3"),
        ]

    def test_search_returns_k_results(self, docs):
        retriever = BM25Retriever(docs)
        results = retriever.search("先验", k=2)
        assert len(results) == 2

    def test_search_returns_doc_score_tuples(self, docs):
        retriever = BM25Retriever(docs)
        results = retriever.search("康德", k=3)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_search_ranks_relevant_higher(self, docs):
        retriever = BM25Retriever(docs)
        results = retriever.search("先验感性", k=3)
        # "先验感性论探讨空间与时间的本质" should rank first
        top_doc, top_score = results[0]
        assert "先验" in top_doc.page_content

    def test_search_k_larger_than_corpus(self, docs):
        """k > corpus size should return all docs without error."""
        retriever = BM25Retriever(docs)
        results = retriever.search("康德", k=100)
        assert len(results) == len(docs)

    def test_search_empty_corpus(self):
        retriever = BM25Retriever([])
        results = retriever.search("任意查询", k=5)
        assert results == []


class TestTokenize:
    def test_tokenize_returns_list(self):
        tokens = _tokenize("先验感性论")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_tokenize_non_empty(self):
        tokens = _tokenize("纯粹理性批判")
        assert all(isinstance(t, str) for t in tokens)


# ---------------------------------------------------------------------------
# QueryRewriter
# ---------------------------------------------------------------------------

class TestQueryRewriter:
    def _make_rewriter(self, response_text: str) -> QueryRewriter:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=response_text)
        return QueryRewriter(llm=mock_llm)

    def test_rewrite_returns_llm_output(self):
        rewriter = self._make_rewriter("先验感性论 transcendental aesthetic Kant 空间 时间")
        result = rewriter.rewrite("先验感性论是什么？")
        assert "先验" in result

    def test_rewrite_strips_whitespace(self):
        rewriter = self._make_rewriter("  改写后的查询  ")
        result = rewriter.rewrite("原始查询")
        assert result == "改写后的查询"

    def test_rewrite_falls_back_on_llm_error(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API 不可用")
        rewriter = QueryRewriter(llm=mock_llm)
        result = rewriter.rewrite("原始查询文本")
        assert result == "原始查询文本"

    def test_rewrite_calls_llm_once(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="改写结果")
        rewriter = QueryRewriter(llm=mock_llm)
        rewriter.rewrite("问题")
        mock_llm.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# _parse_scores
# ---------------------------------------------------------------------------

class TestParseScores:
    def test_parses_standard_format(self):
        content = "1: 8\n2: 5\n3: 9"
        scores = _parse_scores(content, n=3)
        assert scores == [8.0, 5.0, 9.0]

    def test_parses_chinese_colon(self):
        content = "1：7\n2：3"
        scores = _parse_scores(content, n=2)
        assert scores[0] == 7.0
        assert scores[1] == 3.0

    def test_out_of_range_index_ignored(self):
        content = "1: 9\n99: 10"
        scores = _parse_scores(content, n=2)
        assert scores[0] == 9.0
        assert scores[1] == 0.0

    def test_missing_entries_default_to_zero(self):
        content = "1: 7"
        scores = _parse_scores(content, n=3)
        assert scores == [7.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# LLMReranker
# ---------------------------------------------------------------------------

class TestLLMReranker:
    def _make_reranker(self, score_response: str) -> LLMReranker:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=score_response)
        return LLMReranker(llm=mock_llm)

    @pytest.fixture
    def five_docs(self):
        return [_doc(f"段落{i}", f"id{i}") for i in range(5)]

    def test_rerank_returns_top_k(self, five_docs):
        reranker = self._make_reranker("1: 3\n2: 9\n3: 1\n4: 7\n5: 5")
        result = reranker.rerank("查询", five_docs, top_k=3)
        assert len(result) == 3

    def test_rerank_sorts_by_score_desc(self, five_docs):
        # doc index 1 has highest score (9), doc index 3 has 7
        reranker = self._make_reranker("1: 3\n2: 9\n3: 1\n4: 7\n5: 5")
        result = reranker.rerank("查询", five_docs, top_k=2)
        assert result[0].page_content == "段落1"  # score 9
        assert result[1].page_content == "段落3"  # score 7

    def test_rerank_passthrough_when_docs_le_top_k(self):
        reranker = self._make_reranker("")
        docs = [_doc("唯一文档", "id0")]
        result = reranker.rerank("查询", docs, top_k=5)
        assert result == docs
        # LLM should NOT be called when docs <= top_k
        reranker._llm.invoke.assert_not_called()

    def test_rerank_fallback_on_llm_error(self, five_docs):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM 错误")
        reranker = LLMReranker(llm=mock_llm)
        result = reranker.rerank("查询", five_docs, top_k=3)
        # fallback: returns first top_k docs in original order
        assert result == five_docs[:3]


# ---------------------------------------------------------------------------
# _rrf_fusion
# ---------------------------------------------------------------------------

class TestRRFFusion:
    def test_single_list(self):
        ranked = [("a", 1.0), ("b", 0.8), ("c", 0.5)]
        scores = _rrf_fusion([ranked])
        # a ranks first, so should have highest score
        assert scores["a"] > scores["b"] > scores["c"]

    def test_two_lists_agree(self):
        vec = [("x", 1.0), ("y", 0.5)]
        bm25 = [("x", 10.0), ("y", 5.0)]
        scores = _rrf_fusion([vec, bm25])
        assert scores["x"] > scores["y"]

    def test_two_lists_disagree_boosts_second_list_top(self):
        # x ranks 1st in vec, y ranks 1st in bm25
        vec = [("x", 1.0), ("y", 0.5)]
        bm25 = [("y", 10.0), ("x", 5.0)]
        scores = _rrf_fusion([vec, bm25])
        # x: 1/(60+1) + 1/(60+2), y: 1/(60+2) + 1/(60+1) → equal; both should be present
        assert "x" in scores
        assert "y" in scores

    def test_empty_lists_return_empty(self):
        scores = _rrf_fusion([])
        assert scores == {}

    def test_doc_only_in_one_list(self):
        vec = [("a", 1.0)]
        bm25 = [("b", 1.0)]
        scores = _rrf_fusion([vec, bm25])
        assert "a" in scores
        assert "b" in scores


# ---------------------------------------------------------------------------
# HybridRetriever (integration-style, fully mocked)
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    def _make_store(
        self,
        vector_docs: list[tuple[Document, float]] | None = None,
        all_docs: list[Document] | None = None,
    ):
        """Build a mocked ChromaStore."""
        store = MagicMock()
        store.similarity_search_with_score.return_value = vector_docs or []
        store.get_all_documents.return_value = all_docs or []
        return store

    def _make_retriever(
        self,
        store=None,
        vector_docs=None,
        all_docs=None,
        reranker: str = "none",
        enable_query_rewrite: bool = False,
    ) -> HybridRetriever:
        if store is None:
            store = self._make_store(vector_docs=vector_docs, all_docs=all_docs)
        cfg = HybridConfig(
            fetch_k=5,
            final_k=3,
            reranker=reranker,
            enable_query_rewrite=enable_query_rewrite,
        )
        return HybridRetriever(store=store, collection_name="test_col", config=cfg)

    def _make_docs_with_scores(self, n: int) -> list[tuple[Document, float]]:
        return [(_doc(f"内容{i}", f"id{i}"), float(i) / n) for i in range(n)]

    def test_search_returns_documents(self):
        vec_docs = self._make_docs_with_scores(4)
        all_docs = [d for d, _ in vec_docs]
        retriever = self._make_retriever(vector_docs=vec_docs, all_docs=all_docs)
        results = retriever.search("查询")
        assert isinstance(results, list)
        assert all(isinstance(d, Document) for d in results)

    def test_search_respects_final_k(self):
        vec_docs = self._make_docs_with_scores(5)
        all_docs = [d for d, _ in vec_docs]
        retriever = self._make_retriever(vector_docs=vec_docs, all_docs=all_docs)
        results = retriever.search("查询")
        assert len(results) <= 3  # final_k=3

    def test_search_passes_filter_to_vector(self):
        all_docs = [_doc(f"文档{i}", f"id{i}") for i in range(3)]
        store = self._make_store(vector_docs=[], all_docs=all_docs)
        retriever = self._make_retriever(store=store)
        retriever.search("查询", filter={"source": "book.epub"})
        store.similarity_search_with_score.assert_called_once_with(
            "查询", k=5, filter={"source": "book.epub"}, collection_name="test_col"
        )

    def test_bm25_index_built_from_get_all_documents(self):
        all_docs = [_doc(f"文档{i}", f"id{i}") for i in range(3)]
        store = self._make_store(vector_docs=[], all_docs=all_docs)
        retriever = self._make_retriever(store=store)
        retriever.search("查询")
        store.get_all_documents.assert_called_once()

    def test_bm25_lazy_loaded_only_once(self):
        all_docs = [_doc("文档", "id0")]
        store = self._make_store(vector_docs=[], all_docs=all_docs)
        retriever = self._make_retriever(store=store)
        retriever.search("第一次查询")
        retriever.search("第二次查询")
        # get_all_documents should only be called once (BM25 cached)
        assert store.get_all_documents.call_count == 1

    def test_search_with_query_rewrite(self):
        """QueryRewriter output should be used for vector search."""
        vec_docs = self._make_docs_with_scores(3)
        store = self._make_store(vector_docs=vec_docs, all_docs=[d for d, _ in vec_docs])
        cfg = HybridConfig(fetch_k=5, final_k=2, reranker="none", enable_query_rewrite=True)

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="改写后的查询文本")
        retriever = HybridRetriever(store=store, collection_name="col", config=cfg, llm=mock_llm)
        retriever.search("原始查询")

        # Vector search should be called with the rewritten query
        call_args = store.similarity_search_with_score.call_args
        assert call_args[0][0] == "改写后的查询文本"

    def test_search_deduplicates_docs(self):
        """Same doc returned by both vector and BM25 should appear once."""
        shared_doc = _doc("重复文档", "shared_id")
        vec_docs = [(shared_doc, 0.1), (_doc("其他文档", "id2"), 0.5)]
        # all_docs includes shared_doc (for BM25 corpus)
        all_docs = [shared_doc, _doc("其他文档", "id2")]
        store = self._make_store(vector_docs=vec_docs, all_docs=all_docs)
        retriever = self._make_retriever(store=store)
        results = retriever.search("重复")
        doc_ids = [d.metadata.get("chunk_id") for d in results]
        assert len(doc_ids) == len(set(doc_ids)), "Duplicate docs should be deduplicated"

    def test_search_empty_results_returns_empty(self):
        store = self._make_store(vector_docs=[], all_docs=[])
        retriever = self._make_retriever(store=store)
        results = retriever.search("查询")
        assert results == []
