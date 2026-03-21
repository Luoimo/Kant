"""Tests for ReadingPlanAgent — no real API calls, no real ChromaStore."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from backend.agents.reading_plan_agent import ReadingPlanAgent, ReadingPlanResult, plan_node


def _make_doc(text: str, source: str = "sample.pdf", title: str = "Test Book") -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "book_title": title, "section_indices": "1"},
    )


@pytest.fixture
def mock_store():
    docs = [
        _make_doc("第一章：导言", source="data/books/kant.pdf", title="纯粹理性批判"),
    ]
    store = MagicMock()
    store.collection_name = "test_collection"
    store.list_sources.return_value = ["data/books/kant.pdf", "data/books/nietzsche.pdf"]
    store.similarity_search.return_value = docs
    store.similarity_search_with_score.return_value = [(d, 0.9) for d in docs]
    store.get_all_documents.return_value = docs
    return store


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="## 阅读目标\n\n- 完成两周阅读")
    return llm


class TestReadingPlanAgent:
    def test_run_with_book_source(self, mock_store, mock_llm):
        agent = ReadingPlanAgent(store=mock_store, llm=mock_llm)
        result = agent.run(query="帮我制定两周阅读计划", book_source="data/books/kant.pdf")

        assert isinstance(result, ReadingPlanResult)
        assert "阅读" in result.answer
        assert len(result.retrieved_docs) == 1

    def test_run_without_book_source_uses_list_sources(self, mock_store, mock_llm):
        agent = ReadingPlanAgent(store=mock_store, llm=mock_llm)
        result = agent.run(query="给我一个阅读计划")

        assert isinstance(result, ReadingPlanResult)
        mock_store.list_sources.assert_called_once()

    def test_run_handles_list_sources_failure(self, mock_llm):
        store = MagicMock()
        store.collection_name = "test_collection"
        store.list_sources.side_effect = Exception("connection error")
        store.similarity_search.return_value = []
        store.similarity_search_with_score.return_value = []
        store.get_all_documents.return_value = []
        agent = ReadingPlanAgent(store=store, llm=mock_llm)

        # Should not raise
        result = agent.run(query="制定阅读计划")

        assert isinstance(result, ReadingPlanResult)

    def test_run_empty_sources_and_docs(self, mock_llm):
        store = MagicMock()
        store.collection_name = "test_collection"
        store.list_sources.return_value = []
        store.similarity_search.return_value = []
        store.similarity_search_with_score.return_value = []
        store.get_all_documents.return_value = []
        agent = ReadingPlanAgent(store=store, llm=mock_llm)

        result = agent.run(query="制定计划")

        assert isinstance(result, ReadingPlanResult)
        assert result.citations == []


class TestReadingPlanAgentLoadDegradation:
    def test_load_failure_degrades_to_new(self, mock_store, mock_llm):
        failing_storage = MagicMock()
        failing_storage.load.side_effect = ConnectionError("Notion unavailable")

        agent = ReadingPlanAgent(store=mock_store, llm=mock_llm, plan_storage=failing_storage)
        result = agent.run(
            query="修改计划",
            action="edit",
            storage_path="some-page-id",
        )
        assert isinstance(result, ReadingPlanResult)
        assert result.answer


class TestPlanNode:
    def test_plan_node_reads_state_fields(self, mock_store, mock_llm):
        agent = ReadingPlanAgent(store=mock_store, llm=mock_llm)
        state = {
            "plan_query": "制定阅读计划",
            "plan_book_source": "data/books/kant.pdf",
        }

        patch_dict = plan_node(state, agent=agent)

        assert "answer" in patch_dict
        assert "citations" in patch_dict
        assert "retrieved_docs_count" in patch_dict

    def test_plan_node_falls_back_to_user_input(self, mock_store, mock_llm):
        agent = ReadingPlanAgent(store=mock_store, llm=mock_llm)
        state = {"user_input": "帮我制定计划"}

        patch_dict = plan_node(state, agent=agent)

        assert "answer" in patch_dict
