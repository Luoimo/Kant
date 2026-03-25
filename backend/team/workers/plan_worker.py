"""PlanWorker — wraps PlanEditor.run() for chat-driven plan edits."""
from __future__ import annotations

from typing import Any, Literal

from backend.agents.plan_editor import PlanEditor
from backend.team.worker import AgentWorker


class PlanWorker(AgentWorker):
    def __init__(self, plan_editor: PlanEditor) -> None:
        super().__init__(name="plan", role="reading plan")
        self._plan_editor = plan_editor

    def _execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        book_title = payload.get("book_title", "")
        if not book_title:
            return {
                "answer": "当前未打开任何书籍，无法修改计划。",
                "citations": [],
                "retrieved_docs_count": 0,
                "intent": "plan",
            }

        action_raw = payload.get("action", "edit")
        action: Literal["edit", "extend"] = "extend" if action_raw == "extend" else "edit"

        result = self._plan_editor.run(
            query=payload["query"],
            book_title=book_title,
            book_id=payload.get("book_id", ""),
            action=action,
            memory_context=payload.get("memory_context", ""),
        )

        return {
            "answer": result.answer,
            "citations": [],
            "retrieved_docs_count": 0,
            "intent": "plan",
        }

    def invalidate_bm25(self) -> None:
        retriever = getattr(self._plan_editor, "_retriever", None)
        if retriever is not None:
            retriever.invalidate_bm25()
