# _*_ coding:utf-8 _*_
"""
测试共享 fixture。

样本 EPUB 由 ebooklib 在内存中生成，不依赖外部文件。
所有 SectionContent / CleanedSection / TextChunk 均为纯 Python 数据对象。
"""
from __future__ import annotations

import pytest
from ebooklib import epub

from backend.rag.extracter.epub_extractor import (
    SectionContent, BookContent, TOCEntry,
)
from backend.rag.cleaner.text_cleaner import CleanedSection, CleanedBookContent
from backend.rag.chunker.text_chunker import TextChunk, ChunkMeta

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SAMPLE_TITLE = "Test Book"
SAMPLE_AUTHOR = "Test Author"

SECTION_TEXTS = [
    "第一章 纯粹理性批判导言。\n\n人类理性在其知识的某一门类中具有一种特殊命运，"
    "它被种种它无法拒绝的问题所困扰，因为这些问题是由理性本性本身向它提出的。",

    "第二章 先验感性论。\n\n通过直觉，对象被给予我们，并且只有这样，对象才与我们的"
    "思维相关联。直觉只在对象被给予我们时才得以发生。",

    "第三章 先验逻辑论。\n\n我们的知识起源于心灵的两个基本来源：第一个是接受表象的能力，"
    "即感受性；第二个是通过这些表象认识对象的能力，即自发性。",
]

# ---------------------------------------------------------------------------
# EPUB 文件 fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_epub_path(tmp_path_factory):
    """生成含 3 章节 + 嵌套 TOC 的最小 EPUB，供整个测试会话共享。"""
    book = epub.EpubBook()
    book.set_title(SAMPLE_TITLE)
    book.add_author(SAMPLE_AUTHOR)
    book.set_language("zh")

    chapters = []
    chapter_data = [
        ("第一章 导言",
         "<h1>第一章 导言</h1>"
         "<h2>1.1 理性的命运</h2>"
         "<p>人类理性在其知识的某一门类中具有一种特殊命运。</p>"),
        ("第二章 先验感性论",
         "<h1>第二章 先验感性论</h1>"
         "<p>通过直觉，对象被给予我们。</p>"),
        ("第三章 先验逻辑论",
         "<h1>第三章 先验逻辑论</h1>"
         "<p>我们的知识起源于心灵的两个基本来源。</p>"),
    ]
    for i, (ch_title, body) in enumerate(chapter_data):
        c = epub.EpubHtml(title=ch_title, file_name=f"chap{i+1}.xhtml", lang="zh")
        c.content = f"<html><body>{body}</body></html>"
        book.add_item(c)
        chapters.append(c)

    # 嵌套 TOC：第一章含二级小节，验证 level 字段
    book.toc = [
        (epub.Section("第一章 导言"), [
            epub.Link("chap1.xhtml", "第一章 导言", "chap1"),
            epub.Link("chap1.xhtml#sec1", "1.1 理性的命运", "sec1"),
        ]),
        epub.Link("chap2.xhtml", "第二章 先验感性论", "chap2"),
        epub.Link("chap3.xhtml", "第三章 先验逻辑论", "chap3"),
    ]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    # spine 不含 "nav"，避免 section_index 偏移
    book.spine = chapters

    path = tmp_path_factory.mktemp("epub") / "sample.epub"
    epub.write_epub(str(path), book)
    return path


# ---------------------------------------------------------------------------
# SectionContent fixtures（不依赖文件 I/O）
# ---------------------------------------------------------------------------

def _make_section_content(
    section_index: int = 0,
    text: str | None = None,
    blocks: list[dict] | None = None,
    title: str = "",
) -> SectionContent:
    text = text or SECTION_TEXTS[0]
    blocks = blocks or [
        {
            "text": text,
            "block_no": 0,
            "block_type": 0,
            "is_heading": False,
            "heading_text": "",
        }
    ]
    return SectionContent(
        section_index=section_index,
        title=title,
        text=text,
        blocks=blocks,
        source_href=f"chap{section_index + 1}.xhtml",
    )


@pytest.fixture
def sample_section_content():
    return _make_section_content()


@pytest.fixture
def sample_book_content():
    sections = [_make_section_content(i, SECTION_TEXTS[i]) for i in range(3)]
    return BookContent(
        source="data/books/sample.epub",
        total_sections=3,
        metadata={
            "title": SAMPLE_TITLE,
            "author": SAMPLE_AUTHOR,
            "language": "zh",
            "publisher": "",
        },
        sections=sections,
        toc=[
            (1, "第一章 导言", 0),
            (2, "1.1 理性的命运", 0),
            (1, "第二章 先验感性论", 1),
            (1, "第三章 先验逻辑论", 2),
        ],
    )


# ---------------------------------------------------------------------------
# CleanedSection / CleanedBookContent fixtures
# ---------------------------------------------------------------------------

def _make_cleaned_section(
    section_index: int = 0,
    text: str | None = None,
    chapter_title: str = "",
    section_title: str = "",
) -> CleanedSection:
    src = _make_section_content(section_index, text)
    txt = text if text is not None else SECTION_TEXTS[0]
    return CleanedSection(
        section_index=section_index,
        text=txt,
        blocks=[{
            "text": txt,
            "block_no": 0,
            "block_type": 0,
            "is_heading": False,
            "heading_text": "",
            "start_offset": 0,
            "end_offset": len(txt),
        }],
        source_section=src,
        chapter_title=chapter_title,
        section_title=section_title,
    )


@pytest.fixture
def sample_cleaned_section():
    return _make_cleaned_section()


@pytest.fixture
def sample_cleaned_book_content():
    sections = [
        _make_cleaned_section(i, SECTION_TEXTS[i],
                              chapter_title=f"第{i+1}章",
                              section_title=f"第{i+1}章")
        for i in range(3)
    ]
    return CleanedBookContent(
        source="data/books/sample.epub",
        total_sections=3,
        metadata={"title": SAMPLE_TITLE, "author": SAMPLE_AUTHOR},
        sections=sections,
    )


# ---------------------------------------------------------------------------
# TextChunk fixtures
# ---------------------------------------------------------------------------

def make_chunk(
    text: str = "This is a sample chunk for testing purposes.",
    source: str = "sample.epub",
    section_indices: list[int] | None = None,
    chunk_index: int = 0,
) -> TextChunk:
    from backend.rag.chunker.text_chunker import _sha256_id
    return TextChunk(
        chunk_id=_sha256_id(text),
        text=text,
        char_count=len(text),
        metadata=ChunkMeta(
            source=source,
            section_indices=section_indices if section_indices is not None else [0],
            chunk_index=chunk_index,
            book_title=SAMPLE_TITLE,
            author=SAMPLE_AUTHOR,
            chapter_title="",
            section_title="",
        ),
    )


@pytest.fixture
def sample_chunks():
    return [
        make_chunk(f"Chunk number {i} contains important philosophical content.", chunk_index=i)
        for i in range(5)
    ]
