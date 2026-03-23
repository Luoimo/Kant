"""Tests for PlanEditor.generate() — no real API calls, no real ChromaStore."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from backend.agents.plan_editor import PlanEditor


def _make_store(docs: list[Document] | None = None) -> MagicMock:
    store = MagicMock()
    store.collection_name = "test_col"
    store.get_all_documents.return_value = docs or [
        Document(
            page_content="先验感性论内容" * 100,
            metadata={"section_title": "先验感性论", "source": "kant.epub"},
        ),
        Document(
            page_content="先验分析论内容" * 200,
            metadata={"section_title": "先验分析论", "source": "kant.epub"},
        ),
    ]
    return store


def _make_llm(plan_text: str = "## 建议日程\n\n第1周读第1章") -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=plan_text)
    return llm


class TestPlanEditorGenerate:
    def test_generate_returns_markdown_string(self):
        with tempfile.TemporaryDirectory() as d:
            gen = PlanEditor(
                store=_make_store(),
                llm=_make_llm(),
                plan_storage_dir=Path(d),
            )
            content = gen.generate("纯粹理性批判", book_source="kant.epub")
        assert isinstance(content, str)
        assert "纯粹理性批判" in content

    def test_generate_includes_chapter_checkboxes(self):
        with tempfile.TemporaryDirectory() as d:
            gen = PlanEditor(
                store=_make_store(),
                llm=_make_llm(),
                plan_storage_dir=Path(d),
            )
            content = gen.generate("纯粹理性批判", book_source="kant.epub")
        assert "- [ ]" in content
        assert "先验感性论" in content

    def test_generate_writes_plan_file(self):
        with tempfile.TemporaryDirectory() as d:
            gen = PlanEditor(
                store=_make_store(),
                llm=_make_llm(),
                plan_storage_dir=Path(d),
            )
            gen.generate("纯粹理性批判", book_source="kant.epub")
            md_files = list(Path(d).glob("*.md"))
        assert len(md_files) == 1

    def test_generate_with_reading_goal(self):
        with tempfile.TemporaryDirectory() as d:
            gen = PlanEditor(
                store=_make_store(),
                llm=_make_llm(),
                plan_storage_dir=Path(d),
            )
            content = gen.generate(
                "纯粹理性批判",
                book_source="kant.epub",
                reading_goal="重点研究先验感性论",
            )
        assert "重点研究先验感性论" in content

    def test_generate_skips_rag_when_book_not_in_library(self):
        store = _make_store(docs=[])  # empty → book not in library
        with tempfile.TemporaryDirectory() as d:
            gen = PlanEditor(
                store=store,
                llm=_make_llm(),
                plan_storage_dir=Path(d),
            )
            content = gen.generate("不存在的书", book_source="missing.epub")
        assert isinstance(content, str)
        assert len(content) > 10

    def test_generate_idempotent_overwrites_existing(self):
        """Calling generate twice for same book overwrites the file (not appends)."""
        with tempfile.TemporaryDirectory() as d:
            gen = PlanEditor(
                store=_make_store(),
                llm=_make_llm("## 建议日程\n\n第一次生成"),
                plan_storage_dir=Path(d),
            )
            gen.generate("纯粹理性批判", book_source="kant.epub")

            gen2 = PlanEditor(
                store=_make_store(),
                llm=_make_llm("## 建议日程\n\n第二次生成"),
                plan_storage_dir=Path(d),
            )
            gen2.generate("纯粹理性批判", book_source="kant.epub")

            content = list(Path(d).glob("*.md"))[0].read_text(encoding="utf-8")
        assert "第二次生成" in content
        assert "第一次生成" not in content
