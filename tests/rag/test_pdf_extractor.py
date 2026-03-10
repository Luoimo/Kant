# _*_ coding:utf-8 _*_
# @Time:2026/3/10
# @Author:Chloe
# @File:test_pdf_extractor.py
# @Project:Kant
"""
PDFExtractor 单元测试。

覆盖：文件校验、全量提取、分页提取、页码范围边界、
元数据、纯文本提取、逐页迭代、PageContent 结构、
full_text 属性、图片信息收集。
"""
from __future__ import annotations

import pytest

from backend.rag.extracter.pdf_extractor import PDFExtractor, PageContent, PDFContent


# ---------------------------------------------------------------------------
# 文件校验
# ---------------------------------------------------------------------------

class TestFileValidation:

    def test_raises_if_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="不存在"):
            PDFExtractor(tmp_path / "ghost.pdf")

    def test_raises_if_wrong_extension(self, tmp_path):
        txt_file = tmp_path / "doc.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="不支持"):
            PDFExtractor(txt_file)

    def test_accepts_valid_pdf(self, sample_pdf_path):
        extractor = PDFExtractor(sample_pdf_path)
        assert extractor.path == sample_pdf_path


# ---------------------------------------------------------------------------
# page_count 属性
# ---------------------------------------------------------------------------

class TestPageCount:

    def test_returns_correct_count(self, sample_pdf_path):
        assert PDFExtractor(sample_pdf_path).page_count == 3

    def test_matches_extract_total_pages(self, sample_pdf_path):
        ex = PDFExtractor(sample_pdf_path)
        assert ex.page_count == ex.extract().total_pages


# ---------------------------------------------------------------------------
# get_metadata
# ---------------------------------------------------------------------------

class TestGetMetadata:

    def test_returns_dict(self, sample_pdf_path):
        meta = PDFExtractor(sample_pdf_path).get_metadata()
        assert isinstance(meta, dict)

    def test_contains_required_keys(self, sample_pdf_path):
        meta = PDFExtractor(sample_pdf_path).get_metadata()
        for key in ("title", "author", "subject", "keywords",
                    "creator", "producer", "creation_date", "mod_date", "page_count"):
            assert key in meta

    def test_page_count_matches(self, sample_pdf_path):
        meta = PDFExtractor(sample_pdf_path).get_metadata()
        assert meta["page_count"] == 3

    def test_title_and_author_from_pdf(self, sample_pdf_path):
        meta = PDFExtractor(sample_pdf_path).get_metadata()
        assert meta["title"] == "Test Book"
        assert meta["author"] == "Test Author"


# ---------------------------------------------------------------------------
# extract() —— 全量
# ---------------------------------------------------------------------------

class TestExtractFull:

    def test_returns_pdf_content_instance(self, sample_pdf_path):
        content = PDFExtractor(sample_pdf_path).extract()
        assert isinstance(content, PDFContent)

    def test_source_set_correctly(self, sample_pdf_path):
        content = PDFExtractor(sample_pdf_path).extract()
        assert content.source == str(sample_pdf_path)

    def test_total_pages_correct(self, sample_pdf_path):
        content = PDFExtractor(sample_pdf_path).extract()
        assert content.total_pages == 3

    def test_pages_list_length(self, sample_pdf_path):
        content = PDFExtractor(sample_pdf_path).extract()
        assert len(content.pages) == 3

    def test_pages_are_page_content_instances(self, sample_pdf_path):
        content = PDFExtractor(sample_pdf_path).extract()
        for page in content.pages:
            assert isinstance(page, PageContent)

    def test_page_numbers_are_1based(self, sample_pdf_path):
        content = PDFExtractor(sample_pdf_path).extract()
        for i, page in enumerate(content.pages):
            assert page.page_number == i + 1

    def test_metadata_populated(self, sample_pdf_path):
        content = PDFExtractor(sample_pdf_path).extract()
        assert content.metadata["title"] == "Test Book"
        assert content.metadata["author"] == "Test Author"


# ---------------------------------------------------------------------------
# extract() —— 分页范围
# ---------------------------------------------------------------------------

class TestExtractPageRange:

    def test_single_page(self, sample_pdf_path):
        content = PDFExtractor(sample_pdf_path).extract(start_page=2, end_page=2)
        assert len(content.pages) == 1
        assert content.pages[0].page_number == 2

    def test_partial_range(self, sample_pdf_path):
        content = PDFExtractor(sample_pdf_path).extract(start_page=1, end_page=2)
        assert len(content.pages) == 2
        assert content.pages[0].page_number == 1
        assert content.pages[1].page_number == 2

    def test_end_page_none_returns_all(self, sample_pdf_path):
        content = PDFExtractor(sample_pdf_path).extract(start_page=1, end_page=None)
        assert len(content.pages) == 3

    def test_start_page_beyond_total_raises(self, sample_pdf_path):
        with pytest.raises(ValueError):
            PDFExtractor(sample_pdf_path).extract(start_page=10, end_page=None)

    def test_start_page_greater_than_end_raises(self, sample_pdf_path):
        with pytest.raises(ValueError):
            PDFExtractor(sample_pdf_path).extract(start_page=3, end_page=1)

    def test_total_pages_reflects_whole_doc(self, sample_pdf_path):
        # total_pages 记录整个 PDF 页数，即使只提取了部分
        content = PDFExtractor(sample_pdf_path).extract(start_page=1, end_page=1)
        assert content.total_pages == 3


# ---------------------------------------------------------------------------
# PageContent 结构
# ---------------------------------------------------------------------------

class TestPageContentStructure:

    @pytest.fixture(autouse=True)
    def _setup(self, sample_pdf_path):
        content = PDFExtractor(sample_pdf_path).extract()
        self.page = content.pages[0]

    def test_has_text(self):
        assert isinstance(self.page.text, str)
        assert len(self.page.text) > 0

    def test_has_blocks_list(self):
        assert isinstance(self.page.blocks, list)

    def test_blocks_have_required_keys(self):
        for blk in self.page.blocks:
            assert "bbox" in blk
            assert "text" in blk
            assert "block_no" in blk
            assert "block_type" in blk

    def test_bbox_is_4tuple(self):
        for blk in self.page.blocks:
            assert len(blk["bbox"]) == 4

    def test_page_dimensions_positive(self):
        assert self.page.width > 0
        assert self.page.height > 0

    def test_images_list_exists(self):
        assert isinstance(self.page.images, list)

    def test_text_contains_expected_content(self):
        # 样本 PDF 第 1 页含有这段文字（ASCII 文本，PyMuPDF 默认字体不支持中文）
        assert "Critique of Pure Reason" in self.page.text


# ---------------------------------------------------------------------------
# full_text 属性
# ---------------------------------------------------------------------------

class TestFullText:

    def test_full_text_joins_all_pages(self, sample_pdf_path):
        content = PDFExtractor(sample_pdf_path).extract()
        expected = "\n".join(p.text for p in content.pages)
        assert content.full_text == expected

    def test_full_text_contains_content_from_all_pages(self, sample_pdf_path):
        content = PDFExtractor(sample_pdf_path).extract()
        assert "Critique of Pure Reason" in content.full_text
        assert "Transcendental Aesthetic" in content.full_text
        assert "Transcendental Logic" in content.full_text


# ---------------------------------------------------------------------------
# extract_text_only
# ---------------------------------------------------------------------------

class TestExtractTextOnly:

    def test_returns_string(self, sample_pdf_path):
        text = PDFExtractor(sample_pdf_path).extract_text_only()
        assert isinstance(text, str)

    def test_non_empty(self, sample_pdf_path):
        text = PDFExtractor(sample_pdf_path).extract_text_only()
        assert len(text.strip()) > 0

    def test_page_range_respected(self, sample_pdf_path):
        text_p1 = PDFExtractor(sample_pdf_path).extract_text_only(start_page=1, end_page=1)
        text_all = PDFExtractor(sample_pdf_path).extract_text_only()
        assert len(text_p1) < len(text_all)

    def test_contains_expected_text(self, sample_pdf_path):
        text = PDFExtractor(sample_pdf_path).extract_text_only()
        assert "Critique of Pure Reason" in text


# ---------------------------------------------------------------------------
# iter_pages
# ---------------------------------------------------------------------------

class TestIterPages:

    def test_yields_page_content(self, sample_pdf_path):
        pages = list(PDFExtractor(sample_pdf_path).iter_pages())
        for p in pages:
            assert isinstance(p, PageContent)

    def test_count_matches_page_count(self, sample_pdf_path):
        pages = list(PDFExtractor(sample_pdf_path).iter_pages())
        assert len(pages) == 3

    def test_page_numbers_sequential(self, sample_pdf_path):
        pages = list(PDFExtractor(sample_pdf_path).iter_pages())
        for i, p in enumerate(pages):
            assert p.page_number == i + 1

    def test_partial_range(self, sample_pdf_path):
        pages = list(PDFExtractor(sample_pdf_path).iter_pages(start_page=2, end_page=3))
        assert len(pages) == 2
        assert pages[0].page_number == 2

    def test_is_lazy_iterator(self, sample_pdf_path):
        import types
        it = PDFExtractor(sample_pdf_path).iter_pages()
        assert isinstance(it, types.GeneratorType)


# ---------------------------------------------------------------------------
# extract_images 选项
# ---------------------------------------------------------------------------

class TestExtractImages:

    def test_images_empty_by_default(self, sample_pdf_path):
        content = PDFExtractor(sample_pdf_path).extract(extract_images=False)
        for page in content.pages:
            assert page.images == []

    def test_images_collected_when_requested(self, sample_pdf_path):
        # 样本 PDF 没有嵌入图片，所以 images 仍为空，但字段存在且为 list
        content = PDFExtractor(sample_pdf_path).extract(extract_images=True)
        for page in content.pages:
            assert isinstance(page.images, list)
