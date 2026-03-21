"""Tests for RecommendationAgent — no real API calls, no real ChromaStore."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.agents.recommendation_agent import RecommendationAgent, RecommendationResult, recommend_node


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.collection_name = "test_collection"
    store.list_sources.return_value = ["存在与虚无.epub", "手稿.epub"]
    store.list_book_titles.return_value = [
        {"book_title": "存在与虚无", "author": "萨特", "source": "存在与虚无.epub"},
        {"book_title": "1844年经济学哲学手稿", "author": "马克思", "source": "手稿.epub"},
    ]
    return store


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="### 《存在与虚无》（萨特）✅ 已在库\n\n推荐理由：...")
    return llm


class TestRecommendationAgent:
    def test_run_returns_result(self, mock_store, mock_llm):
        agent = RecommendationAgent(store=mock_store, llm=mock_llm)
        result = agent.run(query="推荐几本小众哲学书")

        assert isinstance(result, RecommendationResult)
        assert len(result.answer) > 0
        assert result.citations == []
        assert result.retrieved_docs == []

    def test_run_with_empty_library_still_calls_llm(self, mock_llm):
        store = MagicMock()
        store.list_sources.return_value = []
        store.list_book_titles.return_value = []
        agent = RecommendationAgent(store=store, llm=mock_llm)

        result = agent.run(query="推荐书")

        # 即使书库为空，LLM 仍应被调用并返回推荐
        assert isinstance(result, RecommendationResult)
        assert len(result.answer) > 0
        mock_llm.invoke.assert_called_once()

    def test_library_books_appear_in_prompt(self, mock_store, mock_llm):
        agent = RecommendationAgent(store=mock_store, llm=mock_llm)
        agent.run(query="推荐哲学书")

        call_args = mock_llm.invoke.call_args[0][0]
        user_msg = next(m["content"] for m in call_args if m["role"] == "user")
        # 书库中的书应该出现在发给 LLM 的 prompt 里
        assert "存在与虚无" in user_msg
        assert "1844年经济学哲学手稿" in user_msg

    def test_empty_library_note_in_prompt(self, mock_llm):
        store = MagicMock()
        store.list_sources.return_value = []
        store.list_book_titles.return_value = []
        agent = RecommendationAgent(store=store, llm=mock_llm)
        agent.run(query="推荐书")

        call_args = mock_llm.invoke.call_args[0][0]
        user_msg = next(m["content"] for m in call_args if m["role"] == "user")
        assert "书库目前为空" in user_msg

    def test_current_book_appears_in_prompt(self, mock_store, mock_llm):
        agent = RecommendationAgent(store=mock_store, llm=mock_llm)
        agent.run(query="推荐类似的书", current_book="存在与虚无")

        call_args = mock_llm.invoke.call_args[0][0]
        user_msg = next(m["content"] for m in call_args if m["role"] == "user")
        assert "存在与虚无" in user_msg


class TestRecommendNode:
    def test_recommend_node_reads_state(self, mock_store, mock_llm):
        agent = RecommendationAgent(store=mock_store, llm=mock_llm)
        state = {"recommend_query": "推荐小众书"}

        patch_dict = recommend_node(state, agent=agent)

        assert "answer" in patch_dict
        assert "citations" in patch_dict
        assert patch_dict["retrieved_docs_count"] == 0

    def test_recommend_node_falls_back_to_user_input(self, mock_store, mock_llm):
        agent = RecommendationAgent(store=mock_store, llm=mock_llm)
        state = {"user_input": "给我推荐书"}

        patch_dict = recommend_node(state, agent=agent)

        assert "answer" in patch_dict

    def test_recommend_node_passes_book_source(self, mock_store, mock_llm):
        agent = RecommendationAgent(store=mock_store, llm=mock_llm)
        state = {"user_input": "推荐类似的书", "book_source": "存在与虚无.epub"}

        recommend_node(state, agent=agent)

        call_args = mock_llm.invoke.call_args[0][0]
        user_msg = next(m["content"] for m in call_args if m["role"] == "user")
        assert "存在与虚无.epub" in user_msg
