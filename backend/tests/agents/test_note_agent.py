"""Tests for NoteAgent — process_qa hook only."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from agents.note_agent import NoteAgent, NoteEntry
from langchain_core.messages import AIMessage

_FAKE_QUESTION = "什么是先验统觉？"
_FAKE_ANSWER = "先验统觉是康德哲学的核心概念，指主体在认识对象时，将各种感觉杂多统一起来的先天能力。"
_FAKE_JSON = """```json
{
  "question_summary": "先验统觉的定义",
  "answer_keypoints": ["康德哲学核心", "统杂多为一的先天能力"],
  "followup_questions": ["先验统觉与经验统觉的区别？"],
  "concepts": ["先验统觉", "杂多"]
}
```"""

def _make_llm():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=_FAKE_JSON)
    return llm

class TestProcessQA:
    def test_returns_note_entry(self):
        with patch('agents.note_agent.create_react_agent') as mock_create_agent:
            agent = NoteAgent(notes_dir=Path("."), llm=_make_llm())
            agent._agent_executor = MagicMock()
            
            entry = agent.process_qa(_FAKE_QUESTION, _FAKE_ANSWER, "纯粹理性批判")
            assert isinstance(entry, NoteEntry)
            assert entry.book_title == "纯粹理性批判"
            assert entry.question_summary == "先验统觉的定义"
            assert entry.concepts == ["先验统觉", "杂多"]

    def test_returns_none_when_no_book_title(self):
        agent = NoteAgent(notes_dir=Path("."), llm=_make_llm())
        result = agent.process_qa(_FAKE_QUESTION, _FAKE_ANSWER, "")
        assert result is None

    def test_returns_none_on_llm_failure(self):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("API Error")
        agent = NoteAgent(notes_dir=Path("."), llm=llm)
        result = agent.process_qa(_FAKE_QUESTION, _FAKE_ANSWER, "康德")
        assert result is None

    def test_invokes_agent_executor(self):
        with patch('agents.note_agent.create_react_agent') as mock_create_agent:
            agent = NoteAgent(notes_dir=Path("."), llm=_make_llm())
            mock_executor = MagicMock()
            agent._agent_executor = mock_executor
            
            agent.process_qa(_FAKE_QUESTION, _FAKE_ANSWER, "纯粹理性批判")
            
            # 确保 ReAct agent 被调用
            mock_executor.invoke.assert_called_once()
            
    def test_calls_vector_store_when_provided(self):
        # 此测试原用于测试向量库同步，由于向量库已被移除，此处可保留为占位或删除
        pass
