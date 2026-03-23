"""Tests for NoteAgent — process_qa hook only."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from backend.agents.note_agent import NoteAgent, NoteEntry

_FAKE_QUESTION = "先验感性论中时间和空间为什么是直观形式？"
_FAKE_ANSWER = "时间和空间是主体感知世界的先天框架，不依赖经验，与牛顿绝对时空观根本不同。"
_EXTRACTED = {
    "question_summary": "时空为何是直观形式",
    "answer_keypoints": ["先天框架", "不依赖经验"],
    "followup_questions": ["物自体为何不可知？"],
    "concepts": ["先验感性论", "时间", "空间"],
}


def _make_llm(extracted: dict = _EXTRACTED) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content=json.dumps(extracted, ensure_ascii=False)
    )
    return llm


class TestProcessQA:
    def test_returns_note_entry(self):
        with tempfile.TemporaryDirectory() as d:
            agent = NoteAgent(notes_dir=Path(d), llm=_make_llm())
            entry = agent.process_qa(_FAKE_QUESTION, _FAKE_ANSWER, "纯粹理性批判")

        assert isinstance(entry, NoteEntry)
        assert entry.book_title == "纯粹理性批判"
        assert entry.question_summary == "时空为何是直观形式"
        assert "先验感性论" in entry.concepts

    def test_writes_markdown_file(self):
        with tempfile.TemporaryDirectory() as d:
            agent = NoteAgent(notes_dir=Path(d), llm=_make_llm())
            agent.process_qa(_FAKE_QUESTION, _FAKE_ANSWER, "纯粹理性批判")

            md_files = list(Path(d).glob("*.md"))
            assert len(md_files) == 1
            content = md_files[0].read_text(encoding="utf-8")
            assert "时空为何是直观形式" in content
            assert "先天框架" in content
            assert "物自体为何不可知" in content

    def test_returns_none_when_no_book_title(self):
        with tempfile.TemporaryDirectory() as d:
            agent = NoteAgent(notes_dir=Path(d), llm=_make_llm())
            result = agent.process_qa(_FAKE_QUESTION, _FAKE_ANSWER, "")
        assert result is None

    def test_returns_none_on_llm_failure(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM error")
        with tempfile.TemporaryDirectory() as d:
            agent = NoteAgent(notes_dir=Path(d), llm=llm)
            result = agent.process_qa(_FAKE_QUESTION, _FAKE_ANSWER, "康德")
        assert result is None

    def test_multiple_qa_append_to_same_file(self):
        with tempfile.TemporaryDirectory() as d:
            agent = NoteAgent(notes_dir=Path(d), llm=_make_llm())
            agent.process_qa(_FAKE_QUESTION, _FAKE_ANSWER, "纯粹理性批判")
            agent.process_qa("另一个问题", "另一个答案", "纯粹理性批判")
            assert len(list(Path(d).glob("*.md"))) == 1

    def test_different_books_create_separate_files(self):
        with tempfile.TemporaryDirectory() as d:
            agent = NoteAgent(notes_dir=Path(d), llm=_make_llm())
            agent.process_qa(_FAKE_QUESTION, _FAKE_ANSWER, "纯粹理性批判")
            agent.process_qa("什么是权力意志？", "权力意志是...", "精神现象学")
            assert len(list(Path(d).glob("*.md"))) == 2

    def test_calls_vector_store_when_provided(self):
        mock_vs = MagicMock()
        mock_vs.search_similar.return_value = []
        with tempfile.TemporaryDirectory() as d:
            agent = NoteAgent(notes_dir=Path(d), llm=_make_llm(), note_vector_store=mock_vs)
            agent.process_qa(_FAKE_QUESTION, _FAKE_ANSWER, "纯粹理性批判")
        mock_vs.add_entry.assert_called_once()
        mock_vs.search_similar.assert_called_once()

    def test_cross_book_ref_appears_in_file(self):
        mock_vs = MagicMock()
        mock_vs.search_similar.return_value = [{
            "book_title": "精神现象学",
            "question_summary": "主体性与客体性",
            "date": "2026-03-10",
        }]
        with tempfile.TemporaryDirectory() as d:
            agent = NoteAgent(notes_dir=Path(d), llm=_make_llm(), note_vector_store=mock_vs)
            agent.process_qa(_FAKE_QUESTION, _FAKE_ANSWER, "纯粹理性批判")
            content = list(Path(d).glob("*.md"))[0].read_text(encoding="utf-8")
        assert "精神现象学" in content
        assert "💡 关联" in content
