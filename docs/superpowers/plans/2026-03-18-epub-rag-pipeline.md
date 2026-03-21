# EPUB RAG Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完全替换 PDF RAG 链路为 EPUB 支持，数据模型采用格式无关的中立命名。

**Architecture:** `EpubExtractor` 解析 EPUB 文件得到 `BookContent`（chapters as sections），经 `TextCleaner` 清洗为 `CleanedBookContent`，再由 `TextChunker` 软切分为带完整章节元数据的 `TextChunk` 列表，最终写入 `ChromaStore`。

**Tech Stack:** Python 3.11+, ebooklib, beautifulsoup4/lxml, unstructured, LangChain, ChromaDB

**Spec:** `docs/superpowers/specs/2026-03-18-epub-rag-pipeline-design.md`

---

## File Map

| 操作 | 文件 |
|---|---|
| 新增 | `backend/rag/extracter/epub_extractor.py` |
| 删除 | `backend/rag/extracter/pdf_extractor.py` |
| 删除 | `tests/rag/test_pdf_extractor.py` |
| 修改 | `backend/rag/extracter/__init__.py` |
| 修改 | `backend/rag/cleaner/text_cleaner.py` |
| 修改 | `backend/rag/cleaner/__init__.py` |
| 修改 | `backend/rag/chunker/text_chunker.py` |
| 修改 | `backend/rag/chunker/__init__.py` |
| 修改 | `backend/rag/chroma/chroma_store.py` |
| 修改 | `tests/rag/conftest.py` |
| 新增 | `tests/rag/test_epub_extractor.py` |
| 修改 | `tests/rag/test_text_cleaner.py` |
| 修改 | `tests/rag/test_text_chunker.py` |
| 修改 | `tests/rag/test_chroma_store.py` |
| 修改 | `requirements.txt` |
| 修改 | `scripts/rag_demo.py` |

---

## Task 1: 更新依赖

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 更新 requirements.txt**

将文件内容替换为：

```
langchain-openai>=0.2.0
langchain-core>=0.3.0
langchain-text-splitters>=0.3.0
pydantic-settings>=2.0.0
langgraph>=0.2.0
chromadb>=0.5.0
tiktoken>=0.7.0
unstructured>=0.14.0
ebooklib>=0.18
beautifulsoup4>=4.12
lxml>=5.0

# 测试依赖
pytest>=8.0.0
pytest-cov>=5.0.0
```

（移除 `pymupdf`、`python-docx`，新增 `ebooklib`、`beautifulsoup4`、`lxml`）

- [ ] **Step 2: 安装新依赖**

```bash
pip install ebooklib beautifulsoup4 lxml
```

Expected: 安装成功，无报错。

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: replace pymupdf/python-docx with ebooklib/beautifulsoup4/lxml"
```

---

## Task 2: 创建 epub_extractor.py（数据模型 + EpubExtractor）

**Files:**
- Create: `backend/rag/extracter/epub_extractor.py`
- Create: `tests/rag/test_epub_extractor.py`
- Modify: `tests/rag/conftest.py`

### Step 1：先更新 conftest.py，创建 EPUB 测试 fixture

- [ ] **Step 1: 替换 tests/rag/conftest.py**

```python
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
```

- [ ] **Step 2: 编写 test_epub_extractor.py（先让它编译失败以确认 epub_extractor.py 尚未存在）**

```bash
python -c "from backend.rag.extracter.epub_extractor import EpubExtractor"
```

Expected: `ModuleNotFoundError`（epub_extractor.py 尚未创建）

- [ ] **Step 3: 创建 tests/rag/test_epub_extractor.py**

```python
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
```

- [ ] **Step 4: 确认测试失败（epub_extractor.py 还未创建）**

```bash
cd G:/pycharm/Kant && pytest tests/rag/test_epub_extractor.py -v 2>&1 | head -20
```

Expected: `ImportError` 或 `ModuleNotFoundError`

- [ ] **Step 5: 创建 backend/rag/extracter/epub_extractor.py**

```python
# _*_ coding:utf-8 _*_
# @File: epub_extractor.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# TOCEntry: (层级 1-based, 标题, section_index 0-based)
TOCEntry = tuple[int, str, int]

_HEADING_TAGS = {"h1", "h2", "h3"}
_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3}


def build_section_map(
    toc: list[TOCEntry],
    total_sections: int,
) -> dict[int, tuple[str, str]]:
    """
    根据 TOC 为每个 section 生成 (章标题, 节标题)。

    - 章：level == 1 的最后一个 section_index <= 当前 section 的条目
    - 节：任意 level 的最后一个 section_index <= 当前 section 的条目
    - 无 TOC 或越界条目时返回 ("", "")。
    """
    if not toc:
        return {i: ("", "") for i in range(total_sections)}

    # 过滤越界条目
    entries: list[TOCEntry] = [
        (lvl, title, idx)
        for lvl, title, idx in toc
        if 0 <= idx < total_sections
    ]

    result: dict[int, tuple[str, str]] = {}
    for sec_no in range(total_sections):
        chapter_title = ""
        section_title = ""
        for lvl, title, start_idx in entries:
            if start_idx <= sec_no:
                if lvl == 1:
                    chapter_title = title
                section_title = title
        result[sec_no] = (chapter_title, section_title)
    return result


@dataclass
class SectionContent:
    """单章节解析结果。"""
    section_index: int       # 0-based，spine 顺序
    title: str               # 章节标题（来自首个 <h1>/<h2>，或 TOC）
    text: str                # 纯文本（空章节为 ""）
    blocks: list[dict]       # 文本块列表
    source_href: str         # EPUB 内部 HTML 文件路径（不含 fragment）


@dataclass
class BookContent:
    """整本 EPUB 的解析结果。"""
    source: str
    total_sections: int
    metadata: dict           # title / author / language / publisher
    sections: list[SectionContent] = field(default_factory=list)
    toc: list[TOCEntry] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(s.text for s in self.sections)


class EpubExtractor:
    """
    基于 ebooklib + BeautifulSoup 的 EPUB 解析工具类。

    主要功能：
    - 提取书籍 metadata（title/author/language/publisher）
    - 从 NCX/NAV 提取 TOC，映射为 [(level, title, section_index)]
    - 按 spine 顺序逐章解析 HTML，构建 SectionContent 列表
    - <h1>~<h3> 直接标记为 is_heading=True，无需字号启发式

    示例::

        extractor = EpubExtractor("kant.epub")
        content = extractor.extract()
        print(content.metadata)
        print(content.sections[0].text[:200])
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"EPUB 文件不存在：{self.path}")
        if self.path.suffix.lower() != ".epub":
            raise ValueError(f"不支持的文件格式（需要 .epub）：{self.path.suffix}")

    def extract(self) -> BookContent:
        """解析 EPUB，返回 BookContent。"""
        book = epub.read_epub(str(self.path), options={"ignore_ncx": False})
        href_index = self._build_href_index(book)
        metadata = self._extract_metadata(book)
        toc = self._extract_toc(book, href_index)
        sections = self._extract_sections(book, href_index)

        logger.debug(
            "EpubExtractor: 解析完成 %s  共 %d 章节，TOC 条目 %d",
            self.path.name, len(sections), len(toc),
        )
        return BookContent(
            source=str(self.path),
            total_sections=len(sections),
            metadata=metadata,
            sections=sections,
            toc=toc,
        )

    def get_metadata(self) -> dict:
        """仅读取书籍元数据，不解析正文。"""
        book = epub.read_epub(str(self.path), options={"ignore_ncx": False})
        return self._extract_metadata(book)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _build_href_index(book: epub.EpubBook) -> dict[str, int]:
        """
        按 spine 顺序构建 {href_without_fragment: section_index}。
        只计入 EpubHtml 类型条目，跳过 EpubNcx/EpubNav。
        """
        index: dict[str, int] = {}
        idx = 0
        for item_id, _ in book.spine:
            item = book.get_item_with_id(item_id)
            if isinstance(item, epub.EpubHtml):
                href = item.get_name()
                index[href] = idx
                idx += 1
        return index

    @staticmethod
    def _extract_metadata(book: epub.EpubBook) -> dict:
        def _first(values: list) -> str:
            return str(values[0][0]) if values else ""

        return {
            "title": _first(book.get_metadata("DC", "title")),
            "author": _first(book.get_metadata("DC", "creator")),
            "language": _first(book.get_metadata("DC", "language")),
            "publisher": _first(book.get_metadata("DC", "publisher")),
        }

    @staticmethod
    def _extract_toc(
        book: epub.EpubBook,
        href_index: dict[str, int],
    ) -> list[TOCEntry]:
        """从 NCX/NAV 提取 TOC，返回 [(level, title, section_index)]。"""
        entries: list[TOCEntry] = []

        def _process(items, level: int = 1) -> None:
            for item in items:
                if isinstance(item, epub.Link):
                    href = item.href.split("#")[0]
                    sec_idx = href_index.get(href)
                    if sec_idx is not None:
                        entries.append((level, item.title or "", sec_idx))
                elif isinstance(item, tuple) and len(item) == 2:
                    section, children = item
                    if isinstance(section, epub.Section) and section.href:
                        href = section.href.split("#")[0]
                        sec_idx = href_index.get(href)
                        if sec_idx is not None:
                            entries.append((level, section.title or "", sec_idx))
                    _process(children, level + 1)

        if book.toc:
            _process(book.toc)
        return entries

    @staticmethod
    def _extract_sections(
        book: epub.EpubBook,
        href_index: dict[str, int],
    ) -> list[SectionContent]:
        """按 spine 顺序解析每个 EpubHtml 章节，构建 SectionContent 列表。"""
        sections: list[SectionContent] = []

        for item_id, _ in book.spine:
            item = book.get_item_with_id(item_id)
            if not isinstance(item, epub.EpubHtml):
                continue

            href = item.get_name()
            section_index = href_index.get(href, len(sections))

            soup = BeautifulSoup(item.get_content(), "lxml")

            # 移除不需要的标签
            for tag in soup.find_all(["script", "style", "header", "footer", "img"]):
                tag.decompose()

            blocks: list[dict] = []
            block_no = 0

            body = soup.find("body") or soup
            for tag in body.find_all(["h1", "h2", "h3", "p"]):
                text = tag.get_text(separator=" ", strip=True)
                if not text:
                    continue
                is_heading = tag.name in _HEADING_TAGS
                blocks.append({
                    "text": text,
                    "block_no": block_no,
                    "block_type": 0,
                    "is_heading": is_heading,
                    "heading_text": text if is_heading else "",
                })
                block_no += 1

            # 章节标题：取首个标题块
            section_title = ""
            for blk in blocks:
                if blk["is_heading"]:
                    section_title = blk["heading_text"]
                    break

            full_text = "\n".join(b["text"] for b in blocks)

            sections.append(SectionContent(
                section_index=section_index,
                title=section_title,
                text=full_text,
                blocks=blocks,
                source_href=href,
            ))

        return sections
```

- [ ] **Step 6: 运行 EpubExtractor 测试**

```bash
cd G:/pycharm/Kant && pytest tests/rag/test_epub_extractor.py -v
```

Expected: 所有测试通过（GREEN）

- [ ] **Step 7: 更新 extracter/__init__.py**

```python
# _*_ coding:utf-8 _*_
from .epub_extractor import (
    EpubExtractor,
    SectionContent,
    BookContent,
    TOCEntry,
    build_section_map,
)

__all__ = [
    "EpubExtractor",
    "SectionContent",
    "BookContent",
    "TOCEntry",
    "build_section_map",
]
```

- [ ] **Step 8: Commit**

```bash
git add backend/rag/extracter/epub_extractor.py \
        backend/rag/extracter/__init__.py \
        tests/rag/conftest.py \
        tests/rag/test_epub_extractor.py
git commit -m "feat: add EpubExtractor with BookContent/SectionContent data model"
```

---

## Task 3: 重写 TextCleaner

**Files:**
- Modify: `backend/rag/cleaner/text_cleaner.py`
- Modify: `backend/rag/cleaner/__init__.py`
- Modify: `tests/rag/test_text_cleaner.py`

- [ ] **Step 1: 更新 test_text_cleaner.py 使用新类名（此时测试会因旧类名不存在而失败）**

将文件中所有引用更新：

```python
# 顶部 import 替换为：
from backend.rag.cleaner.text_cleaner import (
    TextCleaner, CleanConfig,
    CleanedSection, CleanedBookContent,
    build_section_map,  # 现在从 epub_extractor 导入，cleaner 重新导出
)
from backend.rag.extracter.epub_extractor import SectionContent, BookContent

# 所有 CleanedPage → CleanedSection
# 所有 CleanedContent → CleanedBookContent
# 所有 PDFContent → BookContent
# 所有 PageContent → SectionContent
# 所有 clean_page → clean_section
# 所有 clean_content 参数类型更新
# remove_headers_footers 相关测试删除（该功能已移除）
```

运行确认失败：

```bash
cd G:/pycharm/Kant && pytest tests/rag/test_text_cleaner.py -v 2>&1 | head -20
```

Expected: `ImportError` 或 attribute errors（CleanedSection 不存在）

- [ ] **Step 2: 完整替换 backend/rag/cleaner/text_cleaner.py**

```python
# _*_ coding:utf-8 _*_
from __future__ import annotations

import copy
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Sequence

from unstructured.cleaners.core import (
    clean,
    clean_bullets,
    clean_extra_whitespace,
    clean_ordered_bullets,
    group_broken_paragraphs,
    replace_unicode_quotes,
)

from backend.rag.extracter.epub_extractor import (
    BookContent,
    SectionContent,
    TOCEntry,
    build_section_map,
)

logger = logging.getLogger(__name__)


@dataclass
class CleanConfig:
    """
    TextCleaner 的清洗行为开关。

    unstructured 选项
    -----------------
    extra_whitespace    : 合并多余空白符
    bullets             : 移除无序列表符号
    dashes              : 统一破折号
    trailing_punctuation: 删除行尾多余标点
    ordered_bullets     : 移除有序列表编号
    unicode_quotes      : 花引号替换为直引号

    自定义选项
    ----------
    fix_hyphenation     : 修复跨行连字符断词
    remove_page_numbers : 删除孤立页码行（纯数字行）
    normalize_unicode   : NFC 规范化
    min_block_chars     : 文本块最少字符数，低于此值视为噪声丢弃
    """
    # unstructured 原生
    extra_whitespace: bool = True
    bullets: bool = True
    dashes: bool = True
    trailing_punctuation: bool = False
    ordered_bullets: bool = False
    unicode_quotes: bool = True

    # 自定义
    fix_hyphenation: bool = True
    remove_page_numbers: bool = True
    normalize_unicode: bool = True
    min_block_chars: int = 3


@dataclass
class CleanedSection:
    """单章节清洗结果，携带 TOC 推导的章节标题。"""
    section_index: int
    text: str
    blocks: list[dict]              # 清洗后文本块（含 start_offset / end_offset）
    source_section: SectionContent  # 原始 SectionContent 引用
    chapter_title: str = ""         # TOC 层级 1 标题
    section_title: str = ""         # TOC 最细层级标题


@dataclass
class CleanedBookContent:
    """整本书的清洗结果。"""
    source: str
    total_sections: int
    metadata: dict
    sections: list[CleanedSection] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections if s.text.strip())


class TextCleaner:
    """
    基于 unstructured 的 EPUB 文本清洗工具类。

    接受 BookContent / SectionContent 作为输入，
    返回 CleanedBookContent / CleanedSection。

    示例::

        content = EpubExtractor("book.epub").extract()
        cleaned = TextCleaner().clean_content(content)
        print(cleaned.full_text[:500])
    """

    def __init__(self, config: CleanConfig | None = None) -> None:
        self.config = config or CleanConfig()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def clean_content(self, content: BookContent) -> CleanedBookContent:
        """清洗整个 BookContent，返回 CleanedBookContent。"""
        section_map = build_section_map(content.toc, content.total_sections)
        cleaned_sections: list[CleanedSection] = []

        for s in content.sections:
            cs = self.clean_section(s)
            chapter, section = section_map.get(s.section_index, ("", ""))
            cleaned_sections.append(CleanedSection(
                section_index=cs.section_index,
                text=cs.text,
                blocks=cs.blocks,
                source_section=cs.source_section,
                chapter_title=chapter,
                section_title=section,
            ))

        logger.debug(
            "TextCleaner: 完成 %s，共 %d 章节",
            content.source, len(cleaned_sections),
        )
        return CleanedBookContent(
            source=content.source,
            total_sections=content.total_sections,
            metadata=copy.copy(content.metadata),
            sections=cleaned_sections,
        )

    def clean_section(self, section: SectionContent) -> CleanedSection:
        """
        清洗单个 SectionContent，返回 CleanedSection。

        注意：chapter_title / section_title 始终为空字符串；
        这两个字段只有通过 clean_content() 调用才能被正确填充。
        """
        cleaned_blocks = self._clean_blocks(section.blocks)

        for blk in cleaned_blocks:
            blk["text"] = self._apply_pipeline(blk["text"])

        # 计算每块在章节文本中的起止偏移
        cur = 0
        for blk in cleaned_blocks:
            t = (blk.get("text") or "").strip()
            blk["start_offset"] = cur
            if t:
                cur += len(t) + 1
            blk["end_offset"] = cur

        block_texts = [b["text"].strip() for b in cleaned_blocks if b["text"].strip()]
        text = "\n".join(block_texts) if block_texts else ""

        return CleanedSection(
            section_index=section.section_index,
            text=text,
            blocks=cleaned_blocks,
            source_section=section,
            chapter_title="",
            section_title="",
        )

    def clean_text(self, text: str) -> str:
        """直接清洗裸字符串。"""
        return self._apply_pipeline(text)

    def clean_sections(self, sections: Sequence[SectionContent]) -> list[CleanedSection]:
        """批量清洗章节列表。"""
        return [self.clean_section(s) for s in sections]

    # ------------------------------------------------------------------
    # 内部管道
    # ------------------------------------------------------------------

    def _apply_pipeline(self, text: str) -> str:
        cfg = self.config

        if cfg.normalize_unicode:
            text = self._normalize_unicode(text)

        if cfg.extra_whitespace:
            text = re.sub(r"\t", " ", text)

        if cfg.fix_hyphenation:
            text = self._fix_hyphenation(text)

        if cfg.remove_page_numbers:
            text = self._remove_page_numbers(text)

        if cfg.unicode_quotes:
            text = (
                text
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201a", "'").replace("\u201b", "'")
            )
            text = replace_unicode_quotes(text)

        text = clean(
            text,
            extra_whitespace=cfg.extra_whitespace,
            dashes=cfg.dashes,
            bullets=cfg.bullets,
            trailing_punctuation=cfg.trailing_punctuation,
            lowercase=False,
        )

        if cfg.ordered_bullets:
            text = clean_ordered_bullets(text)

        text = group_broken_paragraphs(text)

        if cfg.extra_whitespace:
            text = clean_extra_whitespace(text)

        return text.strip()

    def _clean_blocks(self, blocks: list[dict]) -> list[dict]:
        """
        过滤噪声块：
        - 丢弃字符数低于 min_block_chars 的块
        - 丢弃孤立页码行
        注意：EPUB 无页眉页脚问题（EpubExtractor 在 HTML 层已过滤）。
        """
        cfg = self.config
        result: list[dict] = []

        for blk in blocks:
            txt = blk.get("text", "").strip()
            if len(txt) < cfg.min_block_chars:
                continue
            if cfg.remove_page_numbers and self._is_page_number(txt):
                continue
            cleaned_blk = dict(blk)
            cleaned_blk["text"] = clean_extra_whitespace(
                replace_unicode_quotes(txt)
            ) if cfg.extra_whitespace else txt
            result.append(cleaned_blk)

        return result

    # ------------------------------------------------------------------
    # 启发式规则
    # ------------------------------------------------------------------

    _PAGE_NUMBER_RE = re.compile(
        r"""
        ^[\s\-–—]*
        (
            第\s*\d+\s*页
          | \d+
          | [ivxlcdmIVXLCDM]+
        )
        [\s\-–—.]*$
        """,
        re.VERBOSE,
    )

    _HYPHEN_RE = re.compile(r"(\w)-\n(\w)")

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        return unicodedata.normalize("NFC", text)

    def _fix_hyphenation(self, text: str) -> str:
        return self._HYPHEN_RE.sub(r"\1\2", text)

    def _remove_page_numbers(self, text: str) -> str:
        lines = text.splitlines()
        cleaned = [ln for ln in lines if not self._is_page_number(ln)]
        return "\n".join(cleaned)

    def _is_page_number(self, line: str) -> bool:
        return bool(self._PAGE_NUMBER_RE.match(line.strip()))
```

- [ ] **Step 3: 更新 cleaner/__init__.py**

```python
# _*_ coding:utf-8 _*_
from .text_cleaner import (
    TextCleaner, CleanConfig,
    CleanedSection, CleanedBookContent,
)

__all__ = [
    "TextCleaner", "CleanConfig",
    "CleanedSection", "CleanedBookContent",
]
```

- [ ] **Step 4: 完整替换 tests/rag/test_text_cleaner.py**

```python
# _*_ coding:utf-8 _*_
"""TextCleaner 单元测试"""
from __future__ import annotations
import pytest
from backend.rag.cleaner.text_cleaner import (
    TextCleaner, CleanConfig,
    CleanedSection, CleanedBookContent,
)
from backend.rag.extracter.epub_extractor import SectionContent, BookContent


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
        # 块文字少于 min_block_chars 应被过滤
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
        # sample_book_content.toc 中 (1, "第一章 导言", 0)
        cleaned = TextCleaner().clean_content(sample_book_content)
        assert cleaned.sections[0].chapter_title == "第一章 导言"

    def test_section_title_tracks_fine_grained_toc(self, sample_book_content):
        # section 0 的 section_title 应为最后一个 start<=0 的 TOC 条目（level 2）
        cleaned = TextCleaner().clean_content(sample_book_content)
        # toc = [(1,"第一章 导言",0),(2,"1.1 理性的命运",0),(1,"第二章...",1),...]
        assert cleaned.sections[0].section_title == "1.1 理性的命运"

    def test_second_section_chapter_title(self, sample_book_content):
        cleaned = TextCleaner().clean_content(sample_book_content)
        assert cleaned.sections[1].chapter_title == "第二章 先验感性论"

    def test_no_toc_gives_empty_titles(self):
        from tests.rag.conftest import SECTION_TEXTS, _make_section_content
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
```

- [ ] **Step 5: 运行 TextCleaner 测试**

```bash
cd G:/pycharm/Kant && pytest tests/rag/test_text_cleaner.py -v
```

Expected: 所有测试通过

- [ ] **Step 6: Commit**

```bash
git add backend/rag/cleaner/text_cleaner.py \
        backend/rag/cleaner/__init__.py \
        tests/rag/test_text_cleaner.py
git commit -m "refactor: rewrite TextCleaner for EPUB (CleanedSection/CleanedBookContent)"
```

---

## Task 4: 重写 TextChunker

**Files:**
- Modify: `backend/rag/chunker/text_chunker.py`
- Modify: `backend/rag/chunker/__init__.py`
- Modify: `tests/rag/test_text_chunker.py`

- [ ] **Step 1: 完整替换 backend/rag/chunker/text_chunker.py**

对现有文件做以下修改（逐一说明，逻辑不变）：

**1. 更新 import**
```python
# 替换旧的 import
from backend.rag.cleaner.text_cleaner import CleanedBookContent, CleanedSection
```

**2. 删除 `_PAGE_MARKER` 和 `_PAGE_MARKER_RE`**（仅 `_build_chunks_with_page_inference` 使用，该方法也一并删除）

**3. `ChunkConfig` 字段重命名**
```python
@dataclass
class ChunkConfig:
    splitter: Literal["recursive", "token"] = "recursive"
    chunk_size: int = 1024
    chunk_overlap: int = 64
    encoding_name: str = "cl100k_base"
    separators: list[str] = field(default_factory=lambda: [
        "\n\n", "\n", "。", "！", "？", "；",
        ".", "!", "?", ";", " ", "",
    ])
    section_aware: bool = True      # 原 page_aware
    min_chunk_chars: int = 20
```

**4. `ChunkMeta` 字段重命名**
```python
@dataclass
class ChunkMeta:
    source: str
    section_indices: list[int]   # 原 page_numbers
    chunk_index: int
    book_title: str = ""         # 原 pdf_title
    author: str = ""             # 原 pdf_author
    chapter_title: str = ""
    section_title: str = ""
```

**5. `TextChunk.to_dict()` 键名更新**
```python
def to_dict(self) -> dict:
    return {
        "chunk_id": self.chunk_id,
        "text": self.text,
        "char_count": self.char_count,
        "source": self.metadata.source,
        "section_indices": self.metadata.section_indices,
        "chunk_index": self.metadata.chunk_index,
        "book_title": self.metadata.book_title,
        "author": self.metadata.author,
        "chapter_title": self.metadata.chapter_title,
        "section_title": self.metadata.section_title,
    }
```

**6. `TextChunker.chunk_content()` — 路由到新方法名**
```python
def chunk_content(self, content: CleanedBookContent) -> list[TextChunk]:
    if self.config.section_aware:
        chunks = self._chunk_section_aware(content)
    else:
        chunks = self._chunk_fulltext(content)
    logger.debug(
        "TextChunker: %s → %d 个 chunk（%s 模式，section_aware=%s）",
        content.source, len(chunks),
        self.config.splitter, self.config.section_aware,
    )
    return chunks
```

**7. `chunk_section()` 方法（原 `chunk_page()`）**
```python
def chunk_section(
    self,
    section: CleanedSection,
    *,
    source: str = "",
    book_metadata: dict | None = None,
    index_offset: int = 0,
) -> list[TextChunk]:
    if not section.text.strip():
        return []
    meta = book_metadata or {}
    raw_chunks = self._soft_split(
        section.text,
        max_len=self.config.chunk_size,
        overlap=self.config.chunk_overlap,
    )
    section_titles = self._section_titles_for_chunks(
        raw_chunks, section.text, section.blocks
    )
    toc_section = getattr(section, "section_title", "") or ""
    if toc_section:
        section_titles = [t if t else toc_section for t in section_titles]
    return self._build_chunks_with_section_titles(
        raw_chunks,
        section_indices=[section.section_index],
        source=source,
        book_title=meta.get("title", ""),
        author=meta.get("author", ""),
        index_offset=index_offset,
        chapter_title=getattr(section, "chapter_title", "") or "",
        section_titles=section_titles,
    )
```

**8. `chunk_text()` 参数重命名**
```python
def chunk_text(
    self,
    text: str,
    *,
    source: str = "",
    section_indices: list[int] | None = None,
    book_metadata: dict | None = None,
    index_offset: int = 0,
) -> list[TextChunk]:
    meta = book_metadata or {}
    raw_chunks = self._soft_split(
        text,
        max_len=self.config.chunk_size,
        overlap=self.config.chunk_overlap,
    )
    return self._build_chunks(
        raw_chunks,
        section_indices=section_indices or [],
        source=source,
        book_title=meta.get("title", ""),
        author=meta.get("author", ""),
        index_offset=index_offset,
        chapter_title="",
        section_title="",
    )
```

**9. `_chunk_section_aware()` 方法（原 `_chunk_page_aware()`）**
```python
def _chunk_section_aware(self, content: CleanedBookContent) -> list[TextChunk]:
    all_chunks: list[TextChunk] = []
    for section in content.sections:
        new_chunks = self.chunk_section(
            section,
            source=content.source,
            book_metadata=content.metadata,
            index_offset=len(all_chunks),
        )
        all_chunks.extend(new_chunks)
    return all_chunks
```

**10. `_chunk_fulltext()` — 替换 `page` → `section`，`page_number` → `section_index`**
```python
def _chunk_fulltext(self, content: CleanedBookContent) -> list[TextChunk]:
    full_text_parts: list[str] = []
    section_spans: list[tuple[int, int, int]] = []  # (start, end, section_index)
    section_titles: dict[int, tuple[str, str]] = {
        s.section_index: (
            getattr(s, "chapter_title", "") or "",
            getattr(s, "section_title", "") or "",
        )
        for s in content.sections
    }
    cursor = 0

    for section in content.sections:
        txt = section.text or ""
        if not txt.strip():
            continue
        start = cursor
        full_text_parts.append(txt)
        cursor += len(txt)
        section_spans.append((start, cursor, section.section_index))

    full_text = "".join(full_text_parts)
    positions = self._soft_split_positions(
        full_text,
        max_len=self.config.chunk_size,
        overlap=self.config.chunk_overlap,
    )

    result: list[TextChunk] = []
    for start, end in positions:
        chunk_text = full_text[start:end].strip()
        if len(chunk_text) < self.config.min_chunk_chars:
            continue

        sections_in_chunk: list[int] = []
        for s_start, s_end, s_idx in section_spans:
            if not (end <= s_start or start >= s_end):
                sections_in_chunk.append(s_idx)
        sections_in_chunk = sorted(set(sections_in_chunk))

        ch_title, sec_title = section_titles.get(
            sections_in_chunk[0], ("", "")
        ) if sections_in_chunk else ("", "")

        result.append(TextChunk(
            chunk_id=_sha256_id(chunk_text),
            text=chunk_text,
            char_count=len(chunk_text),
            metadata=ChunkMeta(
                source=content.source,
                section_indices=sections_in_chunk,
                chunk_index=len(result),
                book_title=content.metadata.get("title", ""),
                author=content.metadata.get("author", ""),
                chapter_title=ch_title,
                section_title=sec_title,
            ),
        ))

    logger.debug(
        "TextChunker(fulltext): %s → %d 个 chunk（section_aware=%s）",
        content.source, len(result), self.config.section_aware,
    )
    return result
```

**11. `_build_chunks()` 和 `_build_chunks_with_section_titles()` 参数重命名**

将 `pdf_title`/`pdf_author`/`page_numbers` 替换为 `book_title`/`author`/`section_indices`。

**12. 删除 `_build_chunks_with_page_inference()`**（未被调用，清理掉）

- [ ] **Step 2: 更新 chunker/__init__.py**

```python
# _*_ coding:utf-8 _*_
from .text_chunker import TextChunker, ChunkConfig, TextChunk, ChunkMeta

__all__ = ["TextChunker", "ChunkConfig", "TextChunk", "ChunkMeta"]
```

- [ ] **Step 3: 完整替换 tests/rag/test_text_chunker.py**

```python
# _*_ coding:utf-8 _*_
"""TextChunker 单元测试"""
from __future__ import annotations
import pytest
from backend.rag.chunker.text_chunker import (
    TextChunker, ChunkConfig, TextChunk, ChunkMeta,
)
from backend.rag.cleaner.text_cleaner import CleanedSection, CleanedBookContent
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
            # 每个 chunk 应在句末标点处结束或在 max_len 处硬切
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
```

- [ ] **Step 4: 运行 TextChunker 测试**

```bash
cd G:/pycharm/Kant && pytest tests/rag/test_text_chunker.py -v
```

Expected: 所有测试通过

- [ ] **Step 5: Commit**

```bash
git add backend/rag/chunker/text_chunker.py \
        backend/rag/chunker/__init__.py \
        tests/rag/test_text_chunker.py
git commit -m "refactor: rename TextChunker fields/methods for EPUB (section_indices, book_title, etc.)"
```

---

## Task 5: 更新 ChromaStore

**Files:**
- Modify: `backend/rag/chroma/chroma_store.py`
- Modify: `tests/rag/test_chroma_store.py`

- [ ] **Step 1: 更新 tests/rag/test_chroma_store.py**

将所有 `ingest_pdf` 替换为 `ingest`，所有 `pdf_title` 替换为 `book_title`，所有 `page_numbers` 替换为 `section_indices`，所有 `PDFExtractor`/`PDFContent` 引用移除。

- [ ] **Step 2: 更新 backend/rag/chroma/chroma_store.py**

**a. 更新 import：**
```python
from backend.rag.chunker.text_chunker import ChunkConfig, TextChunk, TextChunker
from backend.rag.cleaner.text_cleaner import CleanConfig, TextCleaner
from backend.rag.extracter.epub_extractor import EpubExtractor
```

**b. `ChromaStore.ingest()` 替代 `ingest_pdf()`：**
```python
def ingest(
    self,
    path: str | Path,
    *,
    collection_name: str | None = None,
) -> IngestResult:
    """
    EPUB 全流水线入库：提取 → 清洗 → 切块 → 向量化 → 写入 Chroma。
    """
    path = Path(path)
    logger.info("开始入库流水线：%s", path.name)

    extractor = EpubExtractor(path)
    book_content = extractor.extract()
    logger.debug("  ✓ 提取完成，共 %d 章节", len(book_content.sections))

    cleaner = TextCleaner(self.clean_config)
    cleaned = cleaner.clean_content(book_content)
    logger.debug("  ✓ 清洗完成")

    chunker = TextChunker(self.chunk_config)
    chunks = chunker.chunk_content(cleaned)
    logger.debug("  ✓ 切块完成，共 %d 个 chunk", len(chunks))

    db = self._resolve_db(collection_name)
    result = self._ingest_chunks_to_db(chunks, db, source=str(path))
    logger.info("  ✓ 入库完成：%s", result)
    return result
```

**c. `_chunk_to_document()` 字段名更新：**
```python
@staticmethod
def _chunk_to_document(chunk: TextChunk) -> Document:
    meta = chunk.metadata
    return Document(
        page_content=chunk.text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "char_count": chunk.char_count,
            "source": meta.source,
            "section_indices": _PAGE_SEP.join(str(i) for i in meta.section_indices),
            "chunk_index": meta.chunk_index,
            "book_title": meta.book_title,
            "author": meta.author,
            "chapter_title": getattr(meta, "chapter_title", "") or "",
            "section_title": getattr(meta, "section_title", "") or "",
        },
    )
```

**d. 更新类注释、`IngestConfig` docstring、`ChromaStore` docstring** — 移除所有 "PDF" 字样，改为 "EPUB"。

- [ ] **Step 3: 运行 ChromaStore 测试**

```bash
cd G:/pycharm/Kant && pytest tests/rag/test_chroma_store.py -v
```

Expected: 所有测试通过

- [ ] **Step 4: Commit**

```bash
git add backend/rag/chroma/chroma_store.py \
        tests/rag/test_chroma_store.py
git commit -m "refactor: ChromaStore ingest_pdf→ingest, pdf_title→book_title, page_numbers→section_indices"
```

---

## Task 6: 清理旧文件 + 更新脚本

**Files:**
- Delete: `backend/rag/extracter/pdf_extractor.py`
- Delete: `tests/rag/test_pdf_extractor.py`
- Modify: `scripts/rag_demo.py`

- [ ] **Step 1: 删除旧文件**

```bash
cd G:/pycharm/Kant
rm backend/rag/extracter/pdf_extractor.py
rm tests/rag/test_pdf_extractor.py
```

- [ ] **Step 2: 更新 scripts/rag_demo.py**

将 `*.pdf` glob 改为 `*.epub`，`ingest_pdf` 改为 `ingest`，`page_aware` 改为 `section_aware`，输出字段名更新：

```python
def ingest_books(store: ChromaStore, books_dir: Path) -> None:
    """将 data/books 下所有 EPUB 写入向量库。"""
    epub_paths = sorted(books_dir.glob("*.epub"))
    if not epub_paths:
        print(f"[WARN] 目录中没有找到 EPUB：{books_dir}")
        return
    for path in epub_paths:
        print(f"\n=== Ingest: {path.name} ===")
        result = store.ingest(path)
        print(result)
```

```python
chunk_cfg = ChunkConfig(
    chunk_size=512,
    chunk_overlap=64,
    section_aware=False,   # 原 page_aware
)
```

```python
print(f"sections   : {meta.get('section_indices', '')}")
print(f"title      : {meta.get('book_title')}")
print(f"author     : {meta.get('author')}")
```

- [ ] **Step 3: 确认无残留引用**

```bash
cd G:/pycharm/Kant && grep -r "pdf_extractor\|PDFExtractor\|PDFContent\|PageContent\|CleanedPage\|CleanedContent\|ingest_pdf\|page_aware\|pdf_title\|pdf_author\|page_numbers" backend/ scripts/ tests/ 2>/dev/null | grep -v ".pyc"
```

Expected: 无输出（或仅有注释/字符串中的历史引用）

- [ ] **Step 4: 运行全部测试**

```bash
cd G:/pycharm/Kant && pytest tests/ -v
```

Expected: 所有测试通过，无 skip

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete pdf_extractor.py, update rag_demo.py to EPUB"
```

---

## Task 7: 全链路验证

- [ ] **Step 1: 运行带覆盖率的完整测试**

```bash
cd G:/pycharm/Kant && pytest tests/ --cov=backend --cov-report=term-missing -v
```

Expected: 所有测试通过，`backend/rag/extracter/epub_extractor.py` 覆盖率 > 80%

- [ ] **Step 2: 快速冒烟测试（如有真实 EPUB）**

如果 `data/books/` 下有 `.epub` 文件，运行：

```bash
cd G:/pycharm/Kant && python -c "
from backend.rag.extracter.epub_extractor import EpubExtractor
from pathlib import Path
import glob
epubs = list(Path('data/books').glob('*.epub'))
if epubs:
    c = EpubExtractor(epubs[0]).extract()
    print(f'chapters: {c.total_sections}')
    print(f'toc entries: {len(c.toc)}')
    print(f'first section title: {c.sections[0].title}')
    print(f'first 100 chars: {c.sections[0].text[:100]}')
else:
    print('no epub found, skip smoke test')
"
```

- [ ] **Step 3: 最终 commit**

```bash
git add .
git commit -m "test: add full EPUB RAG pipeline integration verification"
```
