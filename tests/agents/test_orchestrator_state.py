from unittest.mock import MagicMock

from backend.agents.orchestrator_agent import (
    GraphState,
    RequestContext,
    _build_supervisor_tools,
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


def test_graphstate_has_book_id_not_book_source():
    annotations = GraphState.__annotations__
    assert "book_id" in annotations
    assert "book_source" not in annotations


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


# ---------------------------------------------------------------------------
# _build_supervisor_tools — 闭包注入测试
# ---------------------------------------------------------------------------

def _make_deps():
    """构建包含所有 agent mock 的 GraphDeps-like 对象。"""
    deps = MagicMock()

    deepread_result = MagicMock()
    deepread_result.citations = []
    deepread_result.retrieved_docs = []
    deepread_result.answer = "deep answer"
    deps.deepread_agent.run.return_value = deepread_result

    plan_result = MagicMock()
    plan_result.citations = []
    plan_result.retrieved_docs = []
    plan_result.answer = "plan answer"
    deps.plan_agent.run.return_value = plan_result

    recommend_result = MagicMock()
    recommend_result.answer = "recommend answer"
    deps.recommend_agent.run.return_value = recommend_result

    return deps


def test_deepread_tool_uses_closure_book_source():
    deps = _make_deps()
    ctx = RequestContext()
    tools = _build_supervisor_tools(
        deps, ctx, memory_context="", thread_id="t1",
        book_source="data/books/kant.epub", book_title="纯粹理性批判",
    )
    deepread = next(t for t in tools if t.name == "deepread_book")
    deepread.invoke({"query": "什么是先验感性论"})
    deps.deepread_agent.run.assert_called_once_with(
        query="什么是先验感性论",
        book_source="data/books/kant.epub",
        memory_context="",
    )


def test_modify_plan_tool_uses_closure_book_title():
    deps = _make_deps()
    ctx = RequestContext()
    tools = _build_supervisor_tools(
        deps, ctx, memory_context="", thread_id="t1",
        book_source="data/books/kant.epub", book_title="纯粹理性批判",
    )
    plan = next(t for t in tools if t.name == "modify_plan")
    plan.invoke({"query": "增加第三章", "action": "extend"})
    deps.plan_agent.run.assert_called_once_with(
        query="增加第三章",
        book_title="纯粹理性批判",
        action="extend",
        memory_context="",
    )


def test_modify_plan_returns_error_when_no_book_title():
    deps = _make_deps()
    ctx = RequestContext()
    tools = _build_supervisor_tools(
        deps, ctx, memory_context="", thread_id="t1",
        book_source=None, book_title="",
    )
    plan = next(t for t in tools if t.name == "modify_plan")
    result = plan.invoke({"query": "增加第三章"})
    assert "未打开" in result
    deps.plan_agent.run.assert_not_called()


def test_recommend_tool_uses_closure_book_title():
    deps = _make_deps()
    ctx = RequestContext()
    tools = _build_supervisor_tools(
        deps, ctx, memory_context="", thread_id="t1",
        book_source="data/books/kant.epub", book_title="纯粹理性批判",
    )
    rec = next(t for t in tools if t.name == "recommend_books")
    rec.invoke({"query": "推荐相似书", "recommend_type": "similar"})
    deps.recommend_agent.run.assert_called_once_with(
        query="推荐相似书",
        current_book="纯粹理性批判",
        memory_context="",
        recommend_type="similar",
    )


def test_recommend_tool_passes_empty_string_when_no_book():
    deps = _make_deps()
    ctx = RequestContext()
    tools = _build_supervisor_tools(
        deps, ctx, memory_context="", thread_id="t1",
        book_source=None, book_title="",
    )
    rec = next(t for t in tools if t.name == "recommend_books")
    rec.invoke({"query": "推荐好书"})
    call_kwargs = deps.recommend_agent.run.call_args.kwargs
    assert call_kwargs["current_book"] == ""


