# _*_ coding:utf-8 _*_
"""EpubExtractor 单元测试"""
from __future__ import annotations

import pytest
from backend.rag.extracter.epub_extractor import (
    EpubExtractor, BookContent, SectionContent, TOCEntry,
    build_section_map,
)


class TestEpubExtractorMetadata:
    def test_extract_returns_book_content(self, sample_epub_path):
        extractor = EpubExtractor(sample_epub_path)
        result = extractor.extract()
        assert isinstance(result, BookContent)

    def test_metadata_title(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        assert result.metadata["title"] == "Test Book"

    def test_metadata_author(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        assert result.metadata["author"] == "Test Author"

    def test_metadata_language(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        assert result.metadata["language"] == "zh"

    def test_get_metadata_standalone(self, sample_epub_path):
        meta = EpubExtractor(sample_epub_path).get_metadata()
        assert meta["title"] == "Test Book"

    def test_source_path(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        assert result.source == str(sample_epub_path)


class TestEpubExtractorSections:
    def test_section_count(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        assert result.total_sections == 3
        assert len(result.sections) == 3

    def test_section_indices_sequential(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        indices = [s.section_index for s in result.sections]
        assert indices == [0, 1, 2]

    def test_section_text_not_empty(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        for section in result.sections:
            assert section.text.strip() != ""

    def test_section_source_href(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        assert result.sections[0].source_href == "chap1.xhtml"
        assert result.sections[1].source_href == "chap2.xhtml"

    def test_h1_marked_as_heading(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        section = result.sections[0]
        heading_blocks = [b for b in section.blocks if b["is_heading"]]
        assert len(heading_blocks) >= 1
        assert any("导言" in b["heading_text"] for b in heading_blocks)

    def test_h2_marked_as_heading(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        section = result.sections[0]  # 第一章含 <h2>
        heading_blocks = [b for b in section.blocks if b["is_heading"]]
        assert any("1.1" in b["heading_text"] for b in heading_blocks)

    def test_paragraph_not_marked_as_heading(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        section = result.sections[0]
        p_blocks = [b for b in section.blocks if not b["is_heading"]]
        assert len(p_blocks) >= 1

    def test_block_type_always_zero(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        for section in result.sections:
            for blk in section.blocks:
                assert blk["block_type"] == 0


class TestEpubExtractorTOC:
    def test_toc_not_empty(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        assert len(result.toc) > 0

    def test_toc_entry_structure(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        for entry in result.toc:
            level, title, section_idx = entry
            assert isinstance(level, int) and level >= 1
            assert isinstance(title, str) and title
            assert isinstance(section_idx, int) and section_idx >= 0

    def test_toc_level1_entries(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        level1 = [e for e in result.toc if e[0] == 1]
        assert len(level1) >= 3  # 三章

    def test_toc_level2_entries(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        level2 = [e for e in result.toc if e[0] == 2]
        assert len(level2) >= 1  # 1.1 理性的命运

    def test_toc_section_index_valid(self, sample_epub_path):
        result = EpubExtractor(sample_epub_path).extract()
        for _, _, section_idx in result.toc:
            assert 0 <= section_idx < result.total_sections


class TestEpubExtractorEdgeCases:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            EpubExtractor(tmp_path / "nonexistent.epub")

    def test_wrong_extension(self, tmp_path):
        f = tmp_path / "book.pdf"
        f.write_bytes(b"fake")
        with pytest.raises(ValueError):
            EpubExtractor(f)

    def test_empty_chapter_preserved(self, tmp_path):
        """空章节（无 <p>/<h*>）仍应包含在 sections 中，text=""。"""
        from ebooklib import epub as epub_lib
        book = epub_lib.EpubBook()
        book.set_title("Empty Test")
        book.add_author("Author")
        c1 = epub_lib.EpubHtml(title="Empty", file_name="empty.xhtml", lang="zh")
        c1.content = "<html><body><div>  </div></body></html>"
        c2 = epub_lib.EpubHtml(title="Real", file_name="real.xhtml", lang="zh")
        c2.content = "<html><body><p>有内容的章节。</p></body></html>"
        book.add_item(c1)
        book.add_item(c2)
        book.add_item(epub_lib.EpubNcx())
        book.add_item(epub_lib.EpubNav())
        book.spine = [c1, c2]
        path = tmp_path / "empty_chap.epub"
        epub_lib.write_epub(str(path), book)

        result = EpubExtractor(path).extract()
        assert result.total_sections == 2
        assert result.sections[0].text == ""
        assert result.sections[1].text.strip() != ""


class TestBuildSectionMap:
    def test_empty_toc_returns_empty_titles(self):
        result = build_section_map([], total_sections=3)
        assert result == {0: ("", ""), 1: ("", ""), 2: ("", "")}

    def test_chapter_title_assigned(self):
        toc: list[TOCEntry] = [(1, "第一章", 0), (1, "第二章", 1)]
        result = build_section_map(toc, total_sections=2)
        assert result[0][0] == "第一章"
        assert result[1][0] == "第二章"

    def test_section_title_tracks_most_recent_entry(self):
        toc: list[TOCEntry] = [
            (1, "第一章", 0),
            (2, "第一节", 0),
            (1, "第二章", 1),
        ]
        result = build_section_map(toc, total_sections=2)
        # section 0 的 section_title 应为最后一个 start<=0 的条目
        assert result[0][1] == "第一节"
        assert result[1][1] == "第二章"

    def test_entry_beyond_total_sections_skipped(self):
        toc: list[TOCEntry] = [(1, "幽灵章节", 99)]
        result = build_section_map(toc, total_sections=3)
        # 越界条目被跳过，结果全为空
        assert all(v == ("", "") for v in result.values())
