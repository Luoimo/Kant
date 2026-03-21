"""Tests for NoteAgent — no real API calls, no real ChromaStore."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from backend.agents.note_agent import NoteAgent, NoteResult, notes_node
from backend.xai.citation import Citation


def _make_doc(text: str, source: str = "sample.pdf", title: str = "Test Book", page: str = "1") -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "book_title": title, "section_indices": page},
    )


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.similarity_search.return_value = [
        _make_doc("第一章讲述了纯粹理性批判的基本概念。"),
        _make_doc("先验感性论探讨了时间和空间的本质。"),
    ]
    return store


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="## 笔记\n\n- 要点1\n- 要点2")
    return llm


class TestNoteAgentRAGPath:
    def test_run_with_book_source_returns_result(self, mock_store, mock_llm):
        agent = NoteAgent(store=mock_store, llm=mock_llm)
        result = agent.run(query="整理第一章的笔记", book_source="sample.pdf")

        assert isinstance(result, NoteResult)
        assert "笔记" in result.answer
        assert len(result.citations) == 2
        assert len(result.retrieved_docs) == 2
        mock_store.similarity_search.assert_called_once_with(
            "整理第一章的笔记", k=8, filter={"source": "sample.pdf"}
        )

    def test_run_without_book_source_searches_all(self, mock_store, mock_llm):
        agent = NoteAgent(store=mock_store, llm=mock_llm)
        result = agent.run(query="整理读书笔记")

        assert isinstance(result, NoteResult)
        mock_store.similarity_search.assert_called_once_with("整理读书笔记", k=8, filter=None)

    def test_run_empty_docs_returns_fallback(self, mock_llm):
        store = MagicMock()
        store.similarity_search.return_value = []
        agent = NoteAgent(store=store, llm=mock_llm)

        result = agent.run(query="整理笔记", book_source="nonexistent.pdf")

        assert isinstance(result, NoteResult)
        assert "没有检索到" in result.answer
        assert result.citations == []
        assert result.retrieved_docs == []


class TestNoteAgentRawTextPath:
    def test_run_with_raw_text_skips_rag(self, mock_llm):
        store = MagicMock()
        agent = NoteAgent(store=store, llm=mock_llm)

        result = agent.run(query="帮我整理", raw_text="这是一段需要整理的文字内容。")

        assert isinstance(result, NoteResult)
        assert result.citations == []
        assert result.retrieved_docs == []
        # store should NOT be called because raw_text is provided and no book_source
        store.similarity_search.assert_not_called()


class TestNotesNode:
    def test_notes_node_reads_state_fields(self, mock_store, mock_llm):
        agent = NoteAgent(store=mock_store, llm=mock_llm)
        state = {
            "notes_query": "整理笔记",
            "notes_book_source": "sample.pdf",
        }

        patch_dict = notes_node(state, agent=agent)

        assert "answer" in patch_dict
        assert "citations" in patch_dict
        assert "retrieved_docs_count" in patch_dict
        assert isinstance(patch_dict["retrieved_docs_count"], int)

    def test_notes_node_falls_back_to_user_input(self, mock_store, mock_llm):
        agent = NoteAgent(store=mock_store, llm=mock_llm)
        state = {"user_input": "请帮我整理笔记"}

        patch_dict = notes_node(state, agent=agent)

        assert "answer" in patch_dict
