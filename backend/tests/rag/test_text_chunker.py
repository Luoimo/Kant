# _*_ coding:utf-8 _*_
"""TextChunker 单元测试"""
from __future__ import annotations
import pytest
from rag.chunker.text_chunker import (
    TextChunker, ChunkConfig, TextChunk, ChunkMeta,
)
from rag.cleaner.text_cleaner import CleanedSection, CleanedBookContent
from tests.rag.conftest import SECTION_TEXTS, _make_cleaned_section


class TestChunkSection:
    def test_returns_list_of_text_chunks(self, sample_cleaned_section):
        chunks = TextChunker().chunk_section(
            sample_cleaned_section, source="test.epub"
        )
        assert isinstance(chunks, list)
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_empty_section_returns_empty_list(self):
        from backend.rag.extracter.epub_extractor import SectionContent
        empty_section = CleanedSection(
            section_index=0, text="   ", blocks=[],
            source_section=SectionContent(0, "", "", [], "chap1.xhtml"),
        )
        assert TextChunker().chunk_section(empty_section) == []

    def test_section_indices_correct(self, sample_cleaned_section):
        chunks = TextChunker().chunk_section(sample_cleaned_section, source="t.epub")
        for chunk in chunks:
            assert chunk.metadata.section_indices == [sample_cleaned_section.section_index]

    def test_chapter_title_carried_in_metadata(self):
        section = _make_cleaned_section(0, chapter_title="第一章", section_title="第一章")
        chunks = TextChunker().chunk_section(section, source="t.epub")
        assert all(c.metadata.chapter_title == "第一章" for c in chunks)

    def test_toc_section_title_fallback(self):
        """当页内标题检测结果为空时，应回退到 TOC section_title"""
        section = _make_cleaned_section(0, section_title="第一节")
        chunks = TextChunker().chunk_section(section, source="t.epub")
        assert all(c.metadata.section_title == "第一节" for c in chunks)

    def test_chunk_id_is_sha256_prefix(self, sample_cleaned_section):
        chunks = TextChunker().chunk_section(sample_cleaned_section, source="t.epub")
        for chunk in chunks:
            assert len(chunk.chunk_id) == 16
            assert all(c in "0123456789abcdef" for c in chunk.chunk_id)

    def test_chinese_not_split_mid_sentence(self):
        text = "人类理性在其知识的某一门类中具有一种特殊命运。" * 30
        section = _make_cleaned_section(0, text=text)
        cfg = ChunkConfig(chunk_size=50, chunk_overlap=5)
        chunks = TextChunker(cfg).chunk_section(section, source="t.epub")
        for chunk in chunks:
            assert len(chunk.text) <= cfg.chunk_size + cfg.chunk_size // 4 + 5


class TestChunkContent:
    def test_all_sections_chunked(self, sample_cleaned_book_content):
        chunks = TextChunker().chunk_content(sample_cleaned_book_content)
        assert len(chunks) > 0

    def test_chunk_indices_sequential(self, sample_cleaned_book_content):
        chunks = TextChunker().chunk_content(sample_cleaned_book_content)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i

    def test_book_title_in_metadata(self, sample_cleaned_book_content):
        chunks = TextChunker().chunk_content(sample_cleaned_book_content)
        for chunk in chunks:
            assert chunk.metadata.book_title == "Test Book"

    def test_author_in_metadata(self, sample_cleaned_book_content):
        chunks = TextChunker().chunk_content(sample_cleaned_book_content)
        for chunk in chunks:
            assert chunk.metadata.author == "Test Author"

    def test_section_indices_present(self, sample_cleaned_book_content):
        chunks = TextChunker().chunk_content(sample_cleaned_book_content)
        for chunk in chunks:
            assert len(chunk.metadata.section_indices) >= 1


class TestChunkText:
    def test_basic(self):
        text = "人类理性在其知识的某一门类中具有一种特殊命运。" * 5
        chunks = TextChunker().chunk_text(text, source="t.epub")
        assert len(chunks) > 0

    def test_section_indices_passed_through(self):
        text = "测试内容。" * 10
        chunks = TextChunker().chunk_text(
            text, source="t.epub", section_indices=[3]
        )
        for c in chunks:
            assert c.metadata.section_indices == [3]


class TestChunkMeta:
    def test_to_dict_keys(self, sample_cleaned_section):
        chunks = TextChunker().chunk_section(sample_cleaned_section, source="t.epub")
        if chunks:
            d = chunks[0].to_dict()
            assert "section_indices" in d
            assert "book_title" in d
            assert "author" in d
            assert "page_numbers" not in d
            assert "pdf_title" not in d
            assert "pdf_author" not in d


class TestFulltextMode:
    def test_fulltext_chunks_cover_all_sections(self, sample_cleaned_book_content):
        cfg = ChunkConfig(section_aware=False, chunk_size=200)
        chunks = TextChunker(cfg).chunk_content(sample_cleaned_book_content)
        all_indices = set()
        for c in chunks:
            all_indices.update(c.metadata.section_indices)
        assert all_indices == {0, 1, 2}
