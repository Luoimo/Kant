# _*_ coding:utf-8 _*_
"""TextCleaner 单元测试"""
from __future__ import annotations
import pytest
from rag.cleaner.text_cleaner import (
    TextCleaner, CleanConfig,
    CleanedSection, CleanedBookContent,
)
from rag.extracter.epub_extractor import SectionContent, BookContent


class TestCleanSection:
    def test_returns_cleaned_section(self, sample_section_content):
        cs = TextCleaner().clean_section(sample_section_content)
        assert isinstance(cs, CleanedSection)

    def test_section_index_preserved(self, sample_section_content):
        cs = TextCleaner().clean_section(sample_section_content)
        assert cs.section_index == sample_section_content.section_index

    def test_text_not_empty_for_non_empty_section(self, sample_section_content):
        cs = TextCleaner().clean_section(sample_section_content)
        assert cs.text.strip() != ""

    def test_source_section_reference(self, sample_section_content):
        cs = TextCleaner().clean_section(sample_section_content)
        assert cs.source_section is sample_section_content

    def test_chapter_title_empty_when_called_standalone(self, sample_section_content):
        # clean_section 单独调用时 chapter_title 始终为空
        cs = TextCleaner().clean_section(sample_section_content)
        assert cs.chapter_title == ""
        assert cs.section_title == ""

    def test_blocks_have_start_offset(self, sample_section_content):
        cs = TextCleaner().clean_section(sample_section_content)
        for blk in cs.blocks:
            assert "start_offset" in blk
            assert "end_offset" in blk

    def test_noise_blocks_filtered(self):
        section = SectionContent(
            section_index=0, title="",
            text="ab",
            blocks=[
                {"text": "ab", "block_no": 0, "block_type": 0,
                 "is_heading": False, "heading_text": ""},
                {"text": "这是正常内容，足够长。", "block_no": 1, "block_type": 0,
                 "is_heading": False, "heading_text": ""},
            ],
            source_href="chap1.xhtml",
        )
        cs = TextCleaner(CleanConfig(min_block_chars=5)).clean_section(section)
        assert len(cs.blocks) == 1
        assert "正常内容" in cs.blocks[0]["text"]

    def test_page_number_lines_removed(self):
        section = SectionContent(
            section_index=0, title="",
            text="123\n正文内容在这里。",
            blocks=[
                {"text": "123", "block_no": 0, "block_type": 0,
                 "is_heading": False, "heading_text": ""},
                {"text": "正文内容在这里。", "block_no": 1, "block_type": 0,
                 "is_heading": False, "heading_text": ""},
            ],
            source_href="chap1.xhtml",
        )
        cs = TextCleaner(CleanConfig(remove_page_numbers=True)).clean_section(section)
        texts = [b["text"] for b in cs.blocks]
        assert not any(t.strip() == "123" for t in texts)


class TestCleanContent:
    def test_returns_cleaned_book_content(self, sample_book_content):
        cleaned = TextCleaner().clean_content(sample_book_content)
        assert isinstance(cleaned, CleanedBookContent)

    def test_section_count_preserved(self, sample_book_content):
        cleaned = TextCleaner().clean_content(sample_book_content)
        assert len(cleaned.sections) == len(sample_book_content.sections)

    def test_chapter_title_from_toc(self, sample_book_content):
        cleaned = TextCleaner().clean_content(sample_book_content)
        assert cleaned.sections[0].chapter_title == "第一章 导言"

    def test_section_title_tracks_fine_grained_toc(self, sample_book_content):
        cleaned = TextCleaner().clean_content(sample_book_content)
        assert cleaned.sections[0].section_title == "1.1 理性的命运"

    def test_second_section_chapter_title(self, sample_book_content):
        cleaned = TextCleaner().clean_content(sample_book_content)
        assert cleaned.sections[1].chapter_title == "第二章 先验感性论"

    def test_no_toc_gives_empty_titles(self):
        from backend.tests.rag.conftest import SECTION_TEXTS, _make_section_content
        sections = [_make_section_content(i, SECTION_TEXTS[i]) for i in range(2)]
        content = BookContent(
            source="test.epub", total_sections=2,
            metadata={}, sections=sections, toc=[],
        )
        cleaned = TextCleaner().clean_content(content)
        assert cleaned.sections[0].chapter_title == ""
        assert cleaned.sections[0].section_title == ""

    def test_source_preserved(self, sample_book_content):
        cleaned = TextCleaner().clean_content(sample_book_content)
        assert cleaned.source == sample_book_content.source

    def test_full_text_property(self, sample_book_content):
        cleaned = TextCleaner().clean_content(sample_book_content)
        assert len(cleaned.full_text) > 0


class TestCleanConfig:
    def test_default_config_has_no_header_footer_fields(self):
        cfg = CleanConfig()
        assert not hasattr(cfg, "remove_headers_footers")
        assert not hasattr(cfg, "header_footer_lines")
