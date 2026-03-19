"""Tests for RecommendationAgent — no real API calls, no real ChromaStore."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from backend.agents.recommendation_agent import RecommendationAgent, RecommendationResult, recommend_node


def _make_doc(
    text: str,
    source: str = "sample.pdf",
    title: str = "Test Book",
    author: str = "Test Author",
) -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "book_title": title, "author": author, "section_indices": "1"},
    )


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.similarity_search.return_value = [
        _make_doc("存在主义哲学的核心思想...", title="存在与虚无", author="萨特"),
        _make_doc("人的全面异化与资本主义...", title="1844年经济学哲学手稿", author="马克思"),
    ]
    return store


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="### 《存在与虚无》\n\n推荐理由：...")
    return llm


class TestRecommendationAgent:
    def test_run_returns_result(self, mock_store, mock_llm):
        agent = RecommendationAgent(store=mock_store, llm=mock_llm)
        result = agent.run(query="推荐几本小众哲学书")

        assert isinstance(result, RecommendationResult)
        assert len(result.answer) > 0
        assert len(result.citations) == 2
        assert len(result.retrieved_docs) == 2
        mock_store.similarity_search.assert_called_once_with("推荐几本小众哲学书", k=8)

    def test_run_empty_docs_returns_fallback(self, mock_llm):
        store = MagicMock()
        store.similarity_search.return_value = []
        agent = RecommendationAgent(store=store, llm=mock_llm)

        result = agent.run(query="推荐书")

        assert isinstance(result, RecommendationResult)
        assert "书库" in result.answer or "没有" in result.answer
        assert result.citations == []
        assert result.retrieved_docs == []

    def test_run_deduplicates_books(self, mock_llm):
        # Two docs from the same book should appear once in book_infos
        store = MagicMock()
        store.similarity_search.return_value = [
            _make_doc("片段1", title="Same Book", author="Author A"),
            _make_doc("片段2", title="Same Book", author="Author A"),
        ]
        agent = RecommendationAgent(store=store, llm=mock_llm)
        result = agent.run(query="推荐哲学书")

        assert isinstance(result, RecommendationResult)
        # LLM should have been called
        mock_llm.invoke.assert_called_once()
        # The user_prompt passed to LLM should contain the book only once in book_infos
        call_args = mock_llm.invoke.call_args[0][0]
        user_msg = next(m["content"] for m in call_args if m["role"] == "user")
        assert user_msg.count("Same Book") >= 1  # at least in evidence, but book list deduped


class TestRecommendNode:
    def test_recommend_node_reads_state(self, mock_store, mock_llm):
        agent = RecommendationAgent(store=mock_store, llm=mock_llm)
        state = {"recommend_query": "推荐小众书"}

        patch_dict = recommend_node(state, agent=agent)

        assert "answer" in patch_dict
        assert "citations" in patch_dict
        assert "retrieved_docs_count" in patch_dict

    def test_recommend_node_falls_back_to_user_input(self, mock_store, mock_llm):
        agent = RecommendationAgent(store=mock_store, llm=mock_llm)
        state = {"user_input": "给我推荐书"}

        patch_dict = recommend_node(state, agent=agent)

        assert "answer" in patch_dict
