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

    @staticmethod
    def extract_cover(epub_path: str | Path, dest_dir: str | Path, book_id: str) -> str:
        """
        从 EPUB 提取封面图片，保存到 dest_dir/{book_id}.{ext}。

        优先匹配文件名含 'cover' 的图片，其次取第一张图片。
        返回保存路径字符串，找不到封面时返回空字符串。
        """
        book = epub.read_epub(str(epub_path), options={"ignore_ncx": False})

        cover_item = None
        first_image = None
        for item in book.get_items():
            if item.get_type() != ebooklib.ITEM_IMAGE:
                continue
            if first_image is None:
                first_image = item
            if "cover" in (item.get_name() or "").lower():
                cover_item = item
                break

        target = cover_item or first_image
        if target is None:
            return ""

        media_type = getattr(target, "media_type", "") or ""
        if "png" in media_type:
            ext = ".png"
        elif "gif" in media_type:
            ext = ".gif"
        elif "webp" in media_type:
            ext = ".webp"
        else:
            ext = ".jpg"

        dest = Path(dest_dir) / f"{book_id}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(target.get_content())
        return str(dest)

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

        toc = book.toc if isinstance(book.toc, (list, tuple)) else []
        if toc:
            _process(toc)
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
