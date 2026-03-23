"""Tests for RecommendationAgent — no real API calls."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.agents.recommendation_agent import RecommendationAgent, RecommendationResult


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="### 《存在与虚无》（萨特）\n\n推荐理由：...")
    return llm


class TestRecommendationAgent:
    def test_run_returns_result(self, mock_llm):
        agent = RecommendationAgent(llm=mock_llm)
        result = agent.run(query="推荐几本小众哲学书")

        assert isinstance(result, RecommendationResult)
        assert len(result.answer) > 0
        assert result.citations == []
        assert result.retrieved_docs == []

    def test_llm_is_called(self, mock_llm):
        agent = RecommendationAgent(llm=mock_llm)
        agent.run(query="推荐书")
        mock_llm.invoke.assert_called_once()

    def test_current_book_appears_in_prompt(self, mock_llm):
        agent = RecommendationAgent(llm=mock_llm)
        agent.run(query="推荐类似的书", current_book="存在与虚无")

        call_args = mock_llm.invoke.call_args[0][0]
        user_msg = next(m["content"] for m in call_args if m["role"] == "user")
        assert "存在与虚无" in user_msg

    def test_exclusion_note_in_prompt_on_second_turn(self, mock_llm):
        from langchain_core.messages import AIMessage
        prev = [AIMessage(content="### 《纯粹理性批判》（康德）\n\n推荐理由：...")]
        agent = RecommendationAgent(llm=mock_llm)
        agent.run(query="还有别的吗", recommend_messages=prev)

        call_args = mock_llm.invoke.call_args[0][0]
        user_msg = next(m["content"] for m in call_args if m["role"] == "user")
        assert "纯粹理性批判" in user_msg
