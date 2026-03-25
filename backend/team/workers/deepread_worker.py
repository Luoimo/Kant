"""DeepReadWorker — wraps DeepReadAgent + NoteAgent auto-hook."""
from __future__ import annotations

import sys
from dataclasses import asdict
from typing import Any

from backend.agents.deepread_agent import DeepReadAgent
from backend.agents.note_agent import NoteAgent
from backend.storage.book_catalog import book_id_from_source
from backend.team.worker import AgentWorker


class DeepReadWorker(AgentWorker):
    def __init__(self, deepread_agent: DeepReadAgent, note_agent: NoteAgent) -> None:
        super().__init__(name="deepread", role="evidence-based Q&A")
        self._deepread_agent = deepread_agent
        self._note_agent = note_agent

    def _execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._deepread_agent.run(
            query=payload["query"],
            book_source=payload.get("book_source"),
            memory_context=payload.get("memory_context", ""),
        )

        # Auto-hook: process Q&A into notes (mirrors original orchestrator behavior)
        book_title = payload.get("book_title", "")
        if not book_title:
            for doc in result.retrieved_docs:
                t = (doc.metadata or {}).get("book_title", "")
                if t:
                    book_title = t
                    break

        if book_title:
            try:
                book_id = payload.get("book_id", "")
                self._note_agent.process_qa(
                    payload["query"], result.answer, book_title, book_id=book_id
                )
            except Exception as exc:
                print(f"[DeepReadWorker] note hook failed: {exc}", file=sys.stderr)

        return {
            "answer": result.answer,
            "citations": [asdict(c) for c in result.citations],
            "retrieved_docs_count": len(result.retrieved_docs),
            "intent": "deepread",
        }

    def invalidate_bm25(self) -> None:
        # DeepReadAgent creates a fresh HybridRetriever per run() call,
        # so there is no persistent BM25 cache to invalidate here.
        pass
