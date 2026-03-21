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
    docs = [
        _make_doc("第一章讲述了纯粹理性批判的基本概念。"),
        _make_doc("先验感性论探讨了时间和空间的本质。"),
    ]
    store = MagicMock()
    store.collection_name = "test_collection"
    store.similarity_search.return_value = docs
    store.similarity_search_with_score.return_value = [(d, 0.9) for d in docs]
    store.get_all_documents.return_value = docs
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

    def test_run_without_book_source_searches_all(self, mock_store, mock_llm):
        agent = NoteAgent(store=mock_store, llm=mock_llm)
        result = agent.run(query="整理读书笔记")

        assert isinstance(result, NoteResult)
        assert len(result.retrieved_docs) == 2

    def test_run_empty_docs_returns_fallback(self, mock_llm):
        store = MagicMock()
        store.collection_name = "test_collection"
        store.similarity_search.return_value = []
        store.similarity_search_with_score.return_value = []
        store.get_all_documents.return_value = []
        agent = NoteAgent(store=store, llm=mock_llm)

        result = agent.run(query="整理笔记", book_source="nonexistent.pdf")

        assert isinstance(result, NoteResult)
        assert "没有检索到" in result.answer
        assert result.citations == []
        assert result.retrieved_docs == []


class TestNoteAgentRawTextPath:
    def test_run_with_raw_text_skips_rag(self, mock_llm):
        store = MagicMock()
        store.collection_name = "test_collection"
        store.get_all_documents.return_value = []
        store.similarity_search_with_score.return_value = []
        agent = NoteAgent(store=store, llm=mock_llm)

        result = agent.run(query="帮我整理", raw_text="这是一段需要整理的文字内容。")

        assert isinstance(result, NoteResult)
        assert result.citations == []
        assert result.retrieved_docs == []
        # HybridRetriever should NOT be called because raw_text only (no book_source)
        store.similarity_search_with_score.assert_not_called()


class TestNoteAgentLoadDegradation:
    def test_load_failure_degrades_to_new(self, mock_store, mock_llm):
        """If load() raises, agent must not propagate the exception — degrades to 'new'."""
        failing_storage = MagicMock()
        failing_storage.load.side_effect = ConnectionError("Notion unavailable")

        agent = NoteAgent(store=mock_store, llm=mock_llm, note_storage=failing_storage)
        # Should not raise — must return a NoteResult even when load fails
        result = agent.run(
            query="修改笔记",
            action="edit",
            storage_path="some-page-id",
        )
        assert isinstance(result, NoteResult)
        assert result.answer  # some content was generated


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


class TestNotesNodeStorage:
    def _make_state(self, action="new", storage_path=""):
        return {
            "user_input": "整理笔记",
            "action": action,
            "notes_last_output": {"storage_path": storage_path},
            "notes_messages": [],
            "memory_context": "",
        }

    def test_new_action_calls_save(self, mock_store, mock_llm):
        storage = MagicMock()
        storage.save.return_value = "local/note_001.md"
        agent = NoteAgent(store=mock_store, llm=mock_llm, note_storage=storage)
        state = self._make_state(action="new")
        notes_node(state, agent=agent)
        storage.save.assert_called_once()
        storage.update.assert_not_called()

    def test_edit_action_calls_update_when_storage_path_exists(self, mock_store, mock_llm):
        storage = MagicMock()
        storage.load.return_value = "# Old Note"
        agent = NoteAgent(store=mock_store, llm=mock_llm, note_storage=storage)
        state = self._make_state(action="edit", storage_path="existing-page-id")
        notes_node(state, agent=agent)
        storage.update.assert_called_once_with("existing-page-id", mock_llm.invoke.return_value.content)
        storage.save.assert_not_called()

    def test_extend_action_calls_update_when_storage_path_exists(self, mock_store, mock_llm):
        storage = MagicMock()
        storage.load.return_value = "# Old Note"
        agent = NoteAgent(store=mock_store, llm=mock_llm, note_storage=storage)
        state = self._make_state(action="extend", storage_path="existing-page-id")
        notes_node(state, agent=agent)
        storage.update.assert_called_once_with("existing-page-id", mock_llm.invoke.return_value.content)
        storage.save.assert_not_called()

    def test_save_failure_does_not_raise(self, mock_store, mock_llm):
        storage = MagicMock()
        storage.save.side_effect = ConnectionError("Notion down")
        agent = NoteAgent(store=mock_store, llm=mock_llm, note_storage=storage)
        state = self._make_state(action="new")
        result = notes_node(state, agent=agent)
        assert result["notes_last_output"]["storage_path"] == ""
        assert result["answer"]  # answer is still returned
