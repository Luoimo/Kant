# _*_ coding:utf-8 _*_
# @Time:2026/3/10
# @Author:Chloe
# @File:test_text_chunker.py
# @Project:Kant
"""
TextChunker 单元测试。

覆盖：chunk_text 基础行为、chunk_size / overlap / min_chars 约束、
chunk_id 一致性、to_dict 结构、page_aware 模式、fulltext 模式、
token splitter、chunk_index 连续性、chunk_page 偏移量、
chunk_content 全局索引。
"""
from __future__ import annotations

import pytest

from backend.rag.chunker.text_chunker import (
    ChunkConfig,
    ChunkMeta,
    TextChunk,
    TextChunker,
    _sha256_id,
)
from backend.rag.cleaner.text_cleaner import CleanedContent, CleanedPage
from tests.rag.conftest import _make_cleaned_page

# 用于测试的较长文本（>512 字符）
LONG_TEXT = (
    "人类理性在其知识的某一门类中具有一种特殊命运，"
    "它被种种它无法拒绝的问题所困扰，因为这些问题是由理性本性本身向它提出的，"
    "但它又无法回答，因为这些问题超越了人类理性的一切能力。"
) * 10  # ~600 字符


# ---------------------------------------------------------------------------
# chunk_text —— 基础行为
# ---------------------------------------------------------------------------

class TestChunkText:

    def test_returns_list(self):
        chunks = TextChunker().chunk_text(LONG_TEXT)
        assert isinstance(chunks, list)

    def test_non_empty_result_for_long_text(self):
        chunks = TextChunker().chunk_text(LONG_TEXT)
        assert len(chunks) > 0

    def test_all_items_are_text_chunk(self):
        for chunk in TextChunker().chunk_text(LONG_TEXT):
            assert isinstance(chunk, TextChunk)

    def test_empty_text_returns_empty_list(self):
        assert TextChunker().chunk_text("") == []

    def test_short_text_below_min_returns_empty(self):
        cfg = ChunkConfig(min_chunk_chars=100)
        assert TextChunker(cfg).chunk_text("hi") == []

    def test_source_set_in_metadata(self):
        chunks = TextChunker().chunk_text(LONG_TEXT, source="test.pdf")
        assert all(c.metadata.source == "test.pdf" for c in chunks)

    def test_page_numbers_set_in_metadata(self):
        chunks = TextChunker().chunk_text(LONG_TEXT, page_numbers=[5, 6])
        assert all(c.metadata.page_numbers == [5, 6] for c in chunks)

    def test_pdf_metadata_propagated(self):
        meta = {"title": "Critique", "author": "Kant"}
        chunks = TextChunker().chunk_text(LONG_TEXT, pdf_metadata=meta)
        assert all(c.metadata.pdf_title == "Critique" for c in chunks)
        assert all(c.metadata.pdf_author == "Kant" for c in chunks)

    def test_index_offset_applied(self):
        chunks = TextChunker(ChunkConfig(chunk_size=100)).chunk_text(
            LONG_TEXT, index_offset=10
        )
        assert chunks[0].metadata.chunk_index >= 10


# ---------------------------------------------------------------------------
# chunk_size 约束
# ---------------------------------------------------------------------------

class TestChunkSize:

    def test_chunks_not_exceed_size(self):
        cfg = ChunkConfig(chunk_size=150, chunk_overlap=0, min_chunk_chars=1)
        chunks = TextChunker(cfg).chunk_text(LONG_TEXT)
        # 允许 splitter 有少量超出（分隔符），给 10% 余量
        for c in chunks:
            assert c.char_count <= 165, f"chunk 超出大小：{c.char_count}"

    def test_multiple_chunks_for_long_text(self):
        cfg = ChunkConfig(chunk_size=100, chunk_overlap=0)
        chunks = TextChunker(cfg).chunk_text(LONG_TEXT)
        assert len(chunks) > 1

    def test_single_chunk_when_text_fits(self):
        short = "Hello world."
        cfg = ChunkConfig(chunk_size=500, min_chunk_chars=1)
        chunks = TextChunker(cfg).chunk_text(short)
        assert len(chunks) == 1
        assert chunks[0].text == short


# ---------------------------------------------------------------------------
# min_chunk_chars 过滤
# ---------------------------------------------------------------------------

class TestMinChunkChars:

    def test_very_short_chunk_filtered(self):
        cfg = ChunkConfig(chunk_size=10, chunk_overlap=0, min_chunk_chars=15)
        chunks = TextChunker(cfg).chunk_text("abc")
        assert chunks == []

    def test_exact_min_chunk_passes(self):
        text = "a" * 20
        cfg = ChunkConfig(chunk_size=500, min_chunk_chars=20)
        chunks = TextChunker(cfg).chunk_text(text)
        assert len(chunks) == 1

    def test_one_below_min_filtered(self):
        text = "a" * 19
        cfg = ChunkConfig(chunk_size=500, min_chunk_chars=20)
        assert TextChunker(cfg).chunk_text(text) == []


# ---------------------------------------------------------------------------
# chunk_id 一致性
# ---------------------------------------------------------------------------

class TestChunkId:

    def test_same_text_gives_same_id(self):
        # min_chunk_chars=1 确保短文本不被过滤
        text = "identical content"
        cfg = ChunkConfig(chunk_size=500, min_chunk_chars=1)
        c1 = TextChunker(cfg).chunk_text(text)
        c2 = TextChunker(cfg).chunk_text(text)
        assert c1[0].chunk_id == c2[0].chunk_id

    def test_different_text_gives_different_id(self):
        cfg = ChunkConfig(chunk_size=500, min_chunk_chars=1)
        c1 = TextChunker(cfg).chunk_text("text A")
        c2 = TextChunker(cfg).chunk_text("text B")
        assert c1[0].chunk_id != c2[0].chunk_id

    def test_chunk_id_is_16_hex_chars(self):
        cfg = ChunkConfig(chunk_size=500, min_chunk_chars=1)
        chunks = TextChunker(cfg).chunk_text("hello world")
        assert len(chunks[0].chunk_id) == 16
        assert all(c in "0123456789abcdef" for c in chunks[0].chunk_id)

    def test_sha256_id_function_directly(self):
        assert _sha256_id("hello") == _sha256_id("hello")
        assert _sha256_id("hello") != _sha256_id("world")
        assert len(_sha256_id("test")) == 16


# ---------------------------------------------------------------------------
# TextChunk.to_dict
# ---------------------------------------------------------------------------

class TestTextChunkToDict:

    @pytest.fixture
    def chunk(self):
        return TextChunk(
            chunk_id="abc123",
            text="sample text",
            char_count=11,
            metadata=ChunkMeta(
                source="book.pdf",
                page_numbers=[3, 4],
                chunk_index=7,
                pdf_title="Pure Reason",
                pdf_author="Kant",
            ),
        )

    def test_has_all_required_keys(self, chunk):
        d = chunk.to_dict()
        for key in ("chunk_id", "text", "char_count", "source",
                    "page_numbers", "chunk_index", "pdf_title", "pdf_author",
                    "chapter_title", "section_title"):
            assert key in d

    def test_values_correct(self, chunk):
        d = chunk.to_dict()
        assert d["chunk_id"] == "abc123"
        assert d["text"] == "sample text"
        assert d["char_count"] == 11
        assert d["source"] == "book.pdf"
        assert d["page_numbers"] == [3, 4]
        assert d["chunk_index"] == 7
        assert d["pdf_title"] == "Pure Reason"
        assert d["pdf_author"] == "Kant"


# ---------------------------------------------------------------------------
# chunk_index 连续性
# ---------------------------------------------------------------------------

class TestChunkIndexSequential:

    def test_indices_start_at_zero(self):
        chunks = TextChunker(ChunkConfig(chunk_size=100)).chunk_text(LONG_TEXT)
        assert chunks[0].metadata.chunk_index == 0

    def test_indices_are_sequential(self):
        chunks = TextChunker(ChunkConfig(chunk_size=100)).chunk_text(LONG_TEXT)
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_offset_shifts_all_indices(self):
        cfg = ChunkConfig(chunk_size=100)
        chunks = TextChunker(cfg).chunk_text(LONG_TEXT, index_offset=5)
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(5, 5 + len(chunks)))


# ---------------------------------------------------------------------------
# char_count 字段
# ---------------------------------------------------------------------------

class TestCharCount:

    def test_char_count_matches_text_len(self):
        chunks = TextChunker(ChunkConfig(chunk_size=100)).chunk_text(LONG_TEXT)
        for c in chunks:
            assert c.char_count == len(c.text)


# ---------------------------------------------------------------------------
# chunk_page
# ---------------------------------------------------------------------------

class TestChunkPage:

    @pytest.fixture
    def page(self):
        return _make_cleaned_page(page_number=3, text=LONG_TEXT)

    def test_returns_list_of_chunks(self, page):
        chunks = TextChunker().chunk_page(page, source="book.pdf")
        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_page_number_in_metadata(self, page):
        chunks = TextChunker().chunk_page(page, source="book.pdf")
        assert all(3 in c.metadata.page_numbers for c in chunks)

    def test_source_in_metadata(self, page):
        chunks = TextChunker().chunk_page(page, source="mybook.pdf")
        assert all(c.metadata.source == "mybook.pdf" for c in chunks)

    def test_index_offset_applied(self, page):
        chunks = TextChunker(ChunkConfig(chunk_size=100)).chunk_page(
            page, source="x.pdf", index_offset=20
        )
        assert chunks[0].metadata.chunk_index >= 20
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(20, 20 + len(chunks)))

    def test_pdf_metadata_from_dict(self, page):
        meta = {"title": "KpV", "author": "Kant"}
        chunks = TextChunker().chunk_page(page, source="x.pdf", pdf_metadata=meta)
        assert all(c.metadata.pdf_title == "KpV" for c in chunks)

    def test_empty_page_returns_empty(self):
        empty_page = _make_cleaned_page(page_number=1, text="   ")
        assert TextChunker().chunk_page(empty_page, source="x.pdf") == []


# ---------------------------------------------------------------------------
# chunk_content —— page_aware 模式
# ---------------------------------------------------------------------------

class TestChunkContentPageAware:

    def test_returns_list(self, sample_cleaned_content):
        chunks = TextChunker().chunk_content(sample_cleaned_content)
        assert isinstance(chunks, list)

    def test_non_empty_for_non_empty_content(self, sample_cleaned_content):
        chunks = TextChunker().chunk_content(sample_cleaned_content)
        assert len(chunks) > 0

    def test_global_chunk_index_sequential(self, sample_cleaned_content):
        chunks = TextChunker().chunk_content(sample_cleaned_content)
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_each_chunk_has_single_page_number(self, sample_cleaned_content):
        cfg = ChunkConfig(page_aware=True)
        chunks = TextChunker(cfg).chunk_content(sample_cleaned_content)
        for c in chunks:
            assert len(c.metadata.page_numbers) == 1

    def test_page_numbers_span_all_pages(self, sample_cleaned_content):
        cfg = ChunkConfig(page_aware=True)
        chunks = TextChunker(cfg).chunk_content(sample_cleaned_content)
        all_pages = {pn for c in chunks for pn in c.metadata.page_numbers}
        assert all_pages == {1, 2, 3}

    def test_source_matches_content(self, sample_cleaned_content):
        chunks = TextChunker().chunk_content(sample_cleaned_content)
        assert all(c.metadata.source == sample_cleaned_content.source for c in chunks)


# ---------------------------------------------------------------------------
# chunk_content —— fulltext 模式
# ---------------------------------------------------------------------------

class TestChunkContentFulltext:

    def test_returns_chunks(self, sample_cleaned_content):
        cfg = ChunkConfig(page_aware=False, chunk_size=200)
        chunks = TextChunker(cfg).chunk_content(sample_cleaned_content)
        assert len(chunks) > 0

    def test_global_chunk_index_sequential(self, sample_cleaned_content):
        cfg = ChunkConfig(page_aware=False, chunk_size=200)
        chunks = TextChunker(cfg).chunk_content(sample_cleaned_content)
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_page_numbers_non_empty(self, sample_cleaned_content):
        cfg = ChunkConfig(page_aware=False, chunk_size=200)
        chunks = TextChunker(cfg).chunk_content(sample_cleaned_content)
        for c in chunks:
            assert len(c.metadata.page_numbers) >= 1

    def test_no_page_markers_in_text(self, sample_cleaned_content):
        cfg = ChunkConfig(page_aware=False, chunk_size=200)
        chunks = TextChunker(cfg).chunk_content(sample_cleaned_content)
        for c in chunks:
            assert "<<<PAGE:" not in c.text


# ---------------------------------------------------------------------------
# token splitter
# ---------------------------------------------------------------------------

class TestTokenSplitter:

    def test_token_splitter_returns_chunks(self):
        cfg = ChunkConfig(splitter="token", chunk_size=50, chunk_overlap=5)
        chunks = TextChunker(cfg).chunk_text(LONG_TEXT)
        assert len(chunks) > 0

    def test_token_splitter_chunk_type(self):
        cfg = ChunkConfig(splitter="token", chunk_size=50, chunk_overlap=5)
        for chunk in TextChunker(cfg).chunk_text(LONG_TEXT):
            assert isinstance(chunk, TextChunk)

    def test_token_splitter_multiple_chunks(self):
        # min_chunk_chars=1：避免中文 token chunk 因字符数少于 20 被过滤
        # chunk_size=50 tokens：足够产生多个 chunk
        cfg = ChunkConfig(splitter="token", chunk_size=50, chunk_overlap=5, min_chunk_chars=1)
        chunks = TextChunker(cfg).chunk_text(LONG_TEXT)
        assert len(chunks) > 1
