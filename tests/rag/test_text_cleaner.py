# _*_ coding:utf-8 _*_
# @Time:2026/3/10
# @Author:Chloe
# @File:test_text_cleaner.py
# @Project:Kant
"""
TextCleaner 单元测试。

覆盖：clean_text 管道各步骤、clean_page 块过滤逻辑、
clean_content 整体结构、CleanConfig 开关控制、
_is_page_number / _is_header_footer 启发式规则。
"""
from __future__ import annotations

import pytest

from backend.rag.cleaner.text_cleaner import (
    CleanConfig,
    CleanedContent,
    CleanedPage,
    TextCleaner,
)
from backend.rag.extracter.pdf_extractor import PageContent, PDFContent


# ---------------------------------------------------------------------------
# clean_text —— 基础文本管道
# ---------------------------------------------------------------------------

class TestCleanTextPipeline:

    def test_extra_whitespace_collapsed(self):
        result = TextCleaner().clean_text("hello   world\t\there")
        assert "   " not in result
        assert "\t" not in result

    def test_consecutive_blank_lines_reduced(self):
        result = TextCleaner().clean_text("para1\n\n\n\npara2")
        assert result.count("\n\n\n") == 0

    def test_fix_hyphenation_merges_word(self):
        result = TextCleaner().clean_text("philo-\nsophy is great")
        assert "philosophy" in result

    def test_fix_hyphenation_mid_sentence(self):
        result = TextCleaner().clean_text("The con-\ncept of pure reason")
        assert "concept" in result

    def test_remove_standalone_page_number_arabic(self):
        result = TextCleaner().clean_text("Some text\n42\nMore text")
        lines = result.splitlines()
        assert "42" not in lines

    def test_remove_standalone_page_number_chinese(self):
        result = TextCleaner().clean_text("Some text\n第42页\nMore text")
        lines = result.splitlines()
        assert not any("第42页" == ln.strip() for ln in lines)

    def test_remove_standalone_roman_numeral(self):
        result = TextCleaner().clean_text("Preface\nxiv\nChapter 1")
        lines = result.splitlines()
        assert "xiv" not in lines

    def test_page_number_inside_sentence_kept(self):
        result = TextCleaner().clean_text("See page 42 for details.")
        assert "42" in result

    def test_unicode_quotes_replaced(self):
        result = TextCleaner().clean_text("\u201chello\u201d")
        # 花引号必须被替换掉（具体替换字符因 unstructured 版本而异，不做硬编码）
        assert "\u201c" not in result, "LEFT DOUBLE QUOTATION MARK 应被替换"
        assert "\u201d" not in result, "RIGHT DOUBLE QUOTATION MARK 应被替换"

    def test_normalize_unicode_nfc(self):
        # é as NFC vs NFD
        import unicodedata
        nfd = unicodedata.normalize("NFD", "café")
        result = TextCleaner().clean_text(nfd)
        assert result == unicodedata.normalize("NFC", "café")

    def test_output_stripped(self):
        result = TextCleaner().clean_text("   hello world   ")
        assert result == result.strip()

    def test_empty_string_returns_empty(self):
        assert TextCleaner().clean_text("") == ""

    def test_whitespace_only_returns_empty(self):
        assert TextCleaner().clean_text("   \n  \t  ") == ""


# ---------------------------------------------------------------------------
# clean_text —— CleanConfig 开关
# ---------------------------------------------------------------------------

class TestCleanConfigSwitches:

    def test_disable_fix_hyphenation(self):
        cfg = CleanConfig(fix_hyphenation=False)
        result = TextCleaner(cfg).clean_text("philo-\nsophy")
        # 不修复，连字符和换行应保留（经过其他清洗后可能有变化，但 philosophy 不会出现）
        assert "philosophy" not in result

    def test_disable_remove_page_numbers(self):
        cfg = CleanConfig(remove_page_numbers=False)
        result = TextCleaner(cfg).clean_text("text\n42\nmore")
        assert "42" in result

    def test_disable_normalize_unicode(self):
        import unicodedata
        cfg = CleanConfig(normalize_unicode=False)
        nfd = unicodedata.normalize("NFD", "café")
        result = TextCleaner(cfg).clean_text(nfd)
        # 不做 NFC，结果保持 NFD
        assert unicodedata.is_normalized("NFC", result) is False

    def test_disable_unicode_quotes(self):
        cfg = CleanConfig(unicode_quotes=False)
        result = TextCleaner(cfg).clean_text("\u201chello\u201d")
        # 花引号未被替换
        assert "\u201c" in result or "\u201d" in result

    def test_disable_extra_whitespace(self):
        cfg = CleanConfig(extra_whitespace=False)
        result = TextCleaner(cfg).clean_text("hello   world")
        assert "   " in result


# ---------------------------------------------------------------------------
# _is_page_number 启发式
# ---------------------------------------------------------------------------

class TestIsPageNumber:

    @pytest.fixture(autouse=True)
    def _cleaner(self):
        self.cleaner = TextCleaner()

    @pytest.mark.parametrize("line", [
        "42", "  42  ", "- 42 -", "第42页", "第 42 页", "xiv", "XIV", "ii",
    ])
    def test_recognized_as_page_number(self, line):
        assert self.cleaner._is_page_number(line) is True

    @pytest.mark.parametrize("line", [
        "Hello 42 world", "See page 42", "Chapter 3", "", "Introduction",
        "42nd edition", "2024-01-01",
    ])
    def test_not_recognized_as_page_number(self, line):
        assert self.cleaner._is_page_number(line) is False


# ---------------------------------------------------------------------------
# _is_header_footer 启发式
# ---------------------------------------------------------------------------

class TestIsHeaderFooter:

    @pytest.fixture(autouse=True)
    def _cleaner(self):
        self.cleaner = TextCleaner()  # header_footer_lines=1, threshold=20px

    def test_top_block_is_header(self):
        blk = {"bbox": (72, 5, 500, 18), "text": "Header"}
        assert self.cleaner._is_header_footer(blk, page_height=842) is True

    def test_bottom_block_is_footer(self):
        blk = {"bbox": (280, 825, 320, 838), "text": "1"}
        assert self.cleaner._is_header_footer(blk, page_height=842) is True

    def test_middle_block_is_not_header_footer(self):
        blk = {"bbox": (72, 200, 500, 600), "text": "Main content"}
        assert self.cleaner._is_header_footer(blk, page_height=842) is False

    def test_zero_page_height_returns_false(self):
        blk = {"bbox": (72, 5, 500, 18), "text": "Header"}
        assert self.cleaner._is_header_footer(blk, page_height=0) is False

    def test_custom_threshold(self):
        cfg = CleanConfig(header_footer_lines=3)  # threshold = 60px
        cleaner = TextCleaner(cfg)
        blk = {"bbox": (72, 30, 500, 55), "text": "Near top"}
        assert cleaner._is_header_footer(blk, page_height=842) is True


# ---------------------------------------------------------------------------
# clean_page —— 块过滤
# ---------------------------------------------------------------------------

class TestCleanPageBlockFiltering:

    def test_image_blocks_removed(self, page_with_image_blocks):
        result = TextCleaner().clean_page(page_with_image_blocks)
        assert all(b["block_type"] != 1 for b in result.blocks)

    def test_short_blocks_removed(self, page_with_short_blocks):
        result = TextCleaner().clean_page(page_with_short_blocks)
        min_chars = TextCleaner().config.min_block_chars
        assert all(len(b["text"]) >= min_chars for b in result.blocks)

    def test_header_footer_blocks_removed(self, page_with_header_footer):
        result = TextCleaner().clean_page(page_with_header_footer)
        # 结果块中不应出现 bbox 紧贴顶部/底部的块
        for blk in result.blocks:
            y0, y1 = blk["bbox"][1], blk["bbox"][3]
            assert y0 >= 20, f"页眉块未被过滤：{blk}"
            assert y1 <= 822, f"页脚块未被过滤：{blk}"

    def test_valid_blocks_preserved(self, sample_page_content):
        result = TextCleaner().clean_page(sample_page_content)
        assert len(result.blocks) >= 1

    def test_custom_min_block_chars(self, page_with_short_blocks):
        cfg = CleanConfig(min_block_chars=1)
        result = TextCleaner(cfg).clean_page(page_with_short_blocks)
        # min 降到 1，短块 "ab" 应保留
        short_texts = [b["text"] for b in result.blocks if len(b["text"]) < 3]
        assert len(short_texts) > 0


# ---------------------------------------------------------------------------
# clean_page —— 返回值结构
# ---------------------------------------------------------------------------

class TestCleanPageStructure:

    def test_returns_cleaned_page_instance(self, sample_page_content):
        result = TextCleaner().clean_page(sample_page_content)
        assert isinstance(result, CleanedPage)

    def test_page_number_preserved(self, sample_page_content):
        result = TextCleaner().clean_page(sample_page_content)
        assert result.page_number == sample_page_content.page_number

    def test_width_height_preserved(self, sample_page_content):
        result = TextCleaner().clean_page(sample_page_content)
        assert result.width == sample_page_content.width
        assert result.height == sample_page_content.height

    def test_source_page_reference_set(self, sample_page_content):
        result = TextCleaner().clean_page(sample_page_content)
        assert result.source_page is sample_page_content

    def test_text_is_string(self, sample_page_content):
        result = TextCleaner().clean_page(sample_page_content)
        assert isinstance(result.text, str)

    def test_text_not_empty_for_content_page(self, sample_page_content):
        result = TextCleaner().clean_page(sample_page_content)
        assert result.text.strip()


# ---------------------------------------------------------------------------
# clean_pages（批量）
# ---------------------------------------------------------------------------

class TestCleanPages:

    def test_returns_list_of_cleaned_pages(self, sample_pdf_content):
        results = TextCleaner().clean_pages(sample_pdf_content.pages)
        assert len(results) == len(sample_pdf_content.pages)
        assert all(isinstance(r, CleanedPage) for r in results)


# ---------------------------------------------------------------------------
# clean_content
# ---------------------------------------------------------------------------

class TestCleanContent:

    def test_returns_cleaned_content_instance(self, sample_pdf_content):
        result = TextCleaner().clean_content(sample_pdf_content)
        assert isinstance(result, CleanedContent)

    def test_source_preserved(self, sample_pdf_content):
        result = TextCleaner().clean_content(sample_pdf_content)
        assert result.source == sample_pdf_content.source

    def test_total_pages_preserved(self, sample_pdf_content):
        result = TextCleaner().clean_content(sample_pdf_content)
        assert result.total_pages == sample_pdf_content.total_pages

    def test_all_pages_cleaned(self, sample_pdf_content):
        result = TextCleaner().clean_content(sample_pdf_content)
        assert len(result.pages) == len(sample_pdf_content.pages)

    def test_metadata_copied(self, sample_pdf_content):
        result = TextCleaner().clean_content(sample_pdf_content)
        assert result.metadata["title"] == sample_pdf_content.metadata["title"]

    def test_metadata_is_independent_copy(self, sample_pdf_content):
        result = TextCleaner().clean_content(sample_pdf_content)
        result.metadata["title"] = "MODIFIED"
        assert sample_pdf_content.metadata["title"] == "Test Book"


# ---------------------------------------------------------------------------
# CleanedContent.full_text 属性
# ---------------------------------------------------------------------------

class TestCleanedContentFullText:

    def test_full_text_joins_non_empty_pages(self, sample_cleaned_content):
        ft = sample_cleaned_content.full_text
        assert "纯粹理性批判" in ft
        assert "先验感性论" in ft

    def test_full_text_separator_is_double_newline(self, sample_cleaned_content):
        ft = sample_cleaned_content.full_text
        assert "\n\n" in ft

    def test_empty_pages_excluded_from_full_text(self):
        from tests.rag.conftest import _make_cleaned_page
        pages = [
            _make_cleaned_page(1, "Page one content."),
            _make_cleaned_page(2, ""),        # 空页
            _make_cleaned_page(3, "Page three content."),
        ]
        content = CleanedContent(source="x.pdf", total_pages=3, metadata={}, pages=pages)
        assert content.full_text == "Page one content.\n\nPage three content."
