"""Tests for PlanEditor — edit/extend only, no new-plan creation."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from backend.agents.plan_editor import PlanEditor, PlanEditResult

_FAKE_ANSWER = "## 更新后的计划\n\n每天读两章"
_EXISTING_PLAN = "# 《康德》阅读计划\n\n## 章节进度\n\n- [ ] 第1章（约30分钟）"


def _make_store() -> MagicMock:
    store = MagicMock()
    store.collection_name = "test_col"
    store.list_sources.return_value = ["kant.epub"]
    store.get_all_documents.return_value = [
        Document(page_content="内容" * 100, metadata={"chapter_title": "第1章", "source": "kant.epub"})
    ]
    return store


def _make_llm() -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=_FAKE_ANSWER)
    llm.bind_tools.return_value = llm
    return llm


def _make_react_graph(answer: str = _FAKE_ANSWER) -> MagicMock:
    graph = MagicMock()
    graph.invoke.return_value = {"messages": [AIMessage(content=answer)]}
    return graph


class TestPlanEditor:
    def test_run_edit_returns_result(self, tmp_path):
        plan_file = tmp_path / "康德.md"
        plan_file.write_text(_EXISTING_PLAN, encoding="utf-8")

        with patch("langgraph.prebuilt.create_react_agent", return_value=_make_react_graph()):
            agent = PlanEditor(store=_make_store(), llm=_make_llm(), plan_storage_dir=tmp_path)
            result = agent.run(query="把计划改成每天两章", book_title="康德", action="edit")

        assert isinstance(result, PlanEditResult)
        assert result.answer == _FAKE_ANSWER

    def test_run_extend_returns_result(self, tmp_path):
        plan_file = tmp_path / "康德.md"
        plan_file.write_text(_EXISTING_PLAN, encoding="utf-8")

        with patch("langgraph.prebuilt.create_react_agent", return_value=_make_react_graph()):
            agent = PlanEditor(store=_make_store(), llm=_make_llm(), plan_storage_dir=tmp_path)
            result = agent.run(query="增加第二部分的内容", book_title="康德", action="extend")

        assert isinstance(result, PlanEditResult)
        assert result.answer

    def test_run_saves_updated_plan(self, tmp_path):
        plan_file = tmp_path / "康德.md"
        plan_file.write_text(_EXISTING_PLAN, encoding="utf-8")

        with patch("langgraph.prebuilt.create_react_agent", return_value=_make_react_graph()):
            agent = PlanEditor(store=_make_store(), llm=_make_llm(), plan_storage_dir=tmp_path)
            agent.run(query="修改计划", book_title="康德", action="edit")

        assert "更新后的计划" in plan_file.read_text(encoding="utf-8")

    def test_run_no_existing_plan_still_returns_result(self, tmp_path):
        with patch("langgraph.prebuilt.create_react_agent", return_value=_make_react_graph()):
            agent = PlanEditor(store=_make_store(), llm=_make_llm(), plan_storage_dir=tmp_path)
            result = agent.run(query="修改计划", book_title="不存在的书", action="edit")
        assert isinstance(result, PlanEditResult)

    def test_exports(self):
        import backend.agents.plan_editor as m
        assert hasattr(m, "PlanEditor")
        assert hasattr(m, "PlanEditResult")
        assert not hasattr(m, "ReadingPlanAgent")
