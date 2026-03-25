"""NotesWorker — placeholder for future user-initiated structured notes via chat.

The auto-hook (Q&A → notes after deepread) lives in DeepReadWorker.
This worker is reserved for future use when the chat interface gains
explicit note-taking commands (new/edit/extend in structured/qa/timeline formats).
"""
from __future__ import annotations

from typing import Any

from backend.agents.note_agent import NoteAgent
from backend.team.worker import AgentWorker


class NotesWorker(AgentWorker):
    def __init__(self, note_agent: NoteAgent) -> None:
        super().__init__(name="notes", role="note taking")
        self._note_agent = note_agent

    def _execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Not dispatched to in the current system; reserved for future use.
        return {
            "answer": "笔记功能暂未在对话中直接支持，请通过笔记界面操作。",
            "citations": [],
            "retrieved_docs_count": 0,
            "intent": "notes",
        }

    def invalidate_bm25(self) -> None:
        retriever = getattr(self._note_agent, "_retriever", None)
        if retriever is not None:
            retriever.invalidate_bm25()
