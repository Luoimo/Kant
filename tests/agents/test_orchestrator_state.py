from unittest.mock import MagicMock

from backend.agents.orchestrator_agent import (
    GraphState,
    RequestContext,
    _resolve_book_title,
    _title_from_docs,
)
from langchain_core.messages import AIMessage


def test_graphstate_has_core_output_fields():
    annotations = GraphState.__annotations__
    for field in ("answer", "citations", "retrieved_docs_count", "intent"):
        assert field in annotations, f"Missing field: {field}"


def test_graphstate_has_storage_pointer_fields():
    # plan_last_output and plan_progress removed — plan is now managed by REST API
    annotations = GraphState.__annotations__
    assert "plan_last_output" not in annotations
    assert "plan_progress" not in annotations


def test_graphstate_has_reader_mode_fields():
    annotations = GraphState.__annotations__
    assert "active_tab" not in annotations
    assert "selected_text" in annotations
    assert "current_chapter" in annotations


def test_graphstate_has_memory_fields():
    annotations = GraphState.__annotations__
    assert "memory_context" in annotations
    assert "messages" in annotations


def test_request_context_defaults():
    ctx = RequestContext()
    assert ctx.citations == []
    assert ctx.retrieved_docs_count == 0
    assert ctx.intent == "deepread"
    assert not hasattr(ctx, "plan_last_output")
    assert not hasattr(ctx, "plan_progress")


def test_request_context_mutation():
    ctx = RequestContext()
    ctx.citations = ["cite1"]
    ctx.retrieved_docs_count = 5
    ctx.intent = "deepread"
    assert ctx.citations == ["cite1"]
    assert ctx.retrieved_docs_count == 5
    assert ctx.intent == "deepread"


# ---------------------------------------------------------------------------
# _resolve_book_title
# ---------------------------------------------------------------------------

def test_resolve_book_title_returns_matching_title():
    store = MagicMock()
    store.list_book_titles.return_value = [
        {"source": "data/books/kant.epub", "book_title": "纯粹理性批判"},
        {"source": "data/books/hegel.epub", "book_title": "精神现象学"},
    ]
    assert _resolve_book_title("data/books/kant.epub", store) == "纯粹理性批判"


def test_resolve_book_title_returns_empty_when_not_found():
    store = MagicMock()
    store.list_book_titles.return_value = [
        {"source": "data/books/hegel.epub", "book_title": "精神现象学"},
    ]
    assert _resolve_book_title("data/books/kant.epub", store) == ""


def test_resolve_book_title_returns_empty_when_book_source_is_none():
    store = MagicMock()
    assert _resolve_book_title(None, store) == ""
    store.list_book_titles.assert_not_called()


def test_resolve_book_title_returns_empty_on_store_exception():
    store = MagicMock()
    store.list_book_titles.side_effect = Exception("chroma down")
    assert _resolve_book_title("data/books/kant.epub", store) == ""


# ---------------------------------------------------------------------------
# _title_from_docs
# ---------------------------------------------------------------------------

def test_title_from_docs_returns_first_title():
    doc1 = MagicMock()
    doc1.metadata = {"book_title": "纯粹理性批判"}
    doc2 = MagicMock()
    doc2.metadata = {"book_title": "精神现象学"}
    assert _title_from_docs([doc1, doc2]) == "纯粹理性批判"


def test_title_from_docs_skips_docs_without_title():
    doc1 = MagicMock()
    doc1.metadata = {}
    doc2 = MagicMock()
    doc2.metadata = {"book_title": "精神现象学"}
    assert _title_from_docs([doc1, doc2]) == "精神现象学"


def test_title_from_docs_returns_empty_when_no_docs():
    assert _title_from_docs([]) == ""


