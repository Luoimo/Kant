"""Tests for NoteService — pure storage CRUD, no LLM."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from backend.services.note_service import NoteService


def _make_service(tmp_path, vector_store=None) -> NoteService:
    return NoteService(notes_dir=tmp_path, note_vector_store=vector_store)


class TestAppendManual:
    def test_writes_manual_entry(self, tmp_path):
        svc = _make_service(tmp_path)
        svc.append_manual("我觉得先验分析论更难理解。", "纯粹理性批判")

        files = list(tmp_path.glob("*.md"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "手记" in content
        assert "先验分析论" in content

    def test_appends_to_existing_file(self, tmp_path):
        svc = _make_service(tmp_path)
        svc.append_manual("第一条手记", "康德")
        svc.append_manual("第二条手记", "康德")
        assert len(list(tmp_path.glob("*.md"))) == 1

    def test_calls_vector_store(self, tmp_path):
        mock_vs = MagicMock()
        svc = _make_service(tmp_path, vector_store=mock_vs)
        svc.append_manual("手动笔记内容", "康德")
        mock_vs.add_entry.assert_called_once()


class TestGetNoteContent:
    def test_returns_empty_for_nonexistent(self, tmp_path):
        svc = _make_service(tmp_path)
        assert svc.get_note_content("不存在的书") == ""

    def test_returns_file_text(self, tmp_path):
        svc = _make_service(tmp_path)
        svc.append_manual("测试内容", "康德")
        content = svc.get_note_content("康德")
        assert "测试内容" in content


class TestListBooks:
    def test_empty_when_no_notes(self, tmp_path):
        svc = _make_service(tmp_path)
        assert svc.list_books() == []

    def test_returns_book_dicts(self, tmp_path):
        svc = _make_service(tmp_path)
        svc.append_manual("内容", "纯粹理性批判")
        result = svc.list_books()
        assert isinstance(result, list)
        assert all(isinstance(b, dict) for b in result)


class TestGetTimeline:
    def test_delegates_to_vector_store(self, tmp_path):
        mock_vs = MagicMock()
        mock_vs.get_timeline.return_value = {"entries": [], "books": [], "concept_frequency": {}}
        svc = _make_service(tmp_path, vector_store=mock_vs)
        svc.get_timeline()
        mock_vs.get_timeline.assert_called_once_with(None)

    def test_fallback_parses_md_headers(self, tmp_path):
        svc = _make_service(tmp_path)
        svc.append_manual("笔记", "纯粹理性批判")
        # append_manual writes "手记" header, which doesn't match "## date · summary"
        # Write a proper QA-style header manually for the fallback test
        note_file = tmp_path / "纯粹理性批判.md"
        note_file.write_text("## 2026-03-22 12:00 · 时空为何是直观形式\n", encoding="utf-8")

        timeline = svc.get_timeline("纯粹理性批判")
        assert len(timeline["entries"]) == 1
        assert timeline["entries"][0]["question_summary"] == "时空为何是直观形式"

    def test_fallback_filtered_by_book(self, tmp_path):
        svc = _make_service(tmp_path)
        (tmp_path / "康德.md").write_text("## 2026-03-22 12:00 · 问题A\n", encoding="utf-8")
        (tmp_path / "黑格尔.md").write_text("## 2026-03-22 13:00 · 问题B\n", encoding="utf-8")
        timeline = svc.get_timeline("康德")
        assert all(e["book_title"] == "康德" for e in timeline["entries"])
