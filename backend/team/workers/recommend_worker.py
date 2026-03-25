"""RecommendWorker — wraps RecommendationAgent."""
from __future__ import annotations

from typing import Any, Literal

from backend.agents.recommendation_agent import RecommendationAgent
from backend.team.worker import AgentWorker


class RecommendWorker(AgentWorker):
    def __init__(self, recommend_agent: RecommendationAgent) -> None:
        super().__init__(name="recommend", role="book recommendation")
        self._recommend_agent = recommend_agent

    def _execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        recommend_type_raw = payload.get("recommend_type", "discover")
        valid_types = {"discover", "similar", "next", "theme"}
        recommend_type: Literal["discover", "similar", "next", "theme"] = (
            recommend_type_raw if recommend_type_raw in valid_types else "discover"
        )

        result = self._recommend_agent.run(
            query=payload["query"],
            current_book=payload.get("current_book", ""),
            memory_context=payload.get("memory_context", ""),
            recommend_type=recommend_type,
        )

        return {
            "answer": result.answer,
            "citations": [],
            "retrieved_docs_count": 0,
            "intent": "recommend",
        }
