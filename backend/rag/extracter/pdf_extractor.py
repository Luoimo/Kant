# _*_ coding:utf-8 _*_
# @Time:2026/3/10
# @Author:Chloe
# @File:pdf_extractor.py
# @Project:Kant

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # pymupdf

logger = logging.getLogger(__name__)

# PyMuPDF 字体 flags：bit 4 = 粗体
_FONT_FLAG_BOLD = 16
# 标题判定：正文字号放大比例下限（例如 1.2 表示比正文大 20%）
_HEADING_SIZE_RATIO_MIN = 1.2
# 正文字号上限比例（超过则视为“特大”可能为封面等，不强制当标题）
_HEADING_SIZE_RATIO_MAX = 2.0
# 居中判定：块中心与页面中心的水平距离 < 页面宽度 * 此比例 则视为居中
_HEADING_CENTER_TOLERANCE = 0.25
# 段前/段后空白：与相邻块的最小垂直间距（pt）超过此值视为“段前段后空白多”
_HEADING_VERTICAL_GAP_MIN = 8.0
# 标题典型特征：段落短、行数少——最大字符数、最大行数
_HEADING_MAX_CHARS = 120
_HEADING_MAX_LINES = 5

# TOC 条目：(层级 1-based, 标题, 起始页码 1-based)
TOCEntry = tuple[int, str, int]


def build_page_section_map(
    toc: list[TOCEntry],
    total_pages: int,
) -> dict[int, tuple[str, str]]:
    """
    根据目录 TOC 为每一页生成 (章标题, 节标题)。

    - 章：层级为 1 的最后一个起始页 <= 当前页的条目
    - 节：任意层级的最后一个起始页 <= 当前页的条目（通常比章更细）
    - 无 TOC 或某页之前无条目时返回 ("", "")。
    """
    if not toc:
        return {p: ("", "") for p in range(1, total_pages + 1)}

    # 归一化为 (lvl, title, page)，保证 page 有效
    entries: list[tuple[int, str, int]] = []
    for item in toc:
        if len(item) < 3:
            continue
        lvl, title, page = int(item[0]), str(item[1]).strip(), int(item[2])
        if page < 1 or page > total_pages:
            continue
        entries.append((lvl, title, page))

    result: dict[int, tuple[str, str]] = {}
    for page_no in range(1, total_pages + 1):
        chapter_title = ""
        section_title = ""
        for lvl, title, start_page in entries:
            if start_page <= page_no:
                if lvl == 1:
                    chapter_title = title
                section_title = title
        result[page_no] = (chapter_title, section_title)
    return result


@dataclass
class PageContent:
    """单页解析结果"""
    page_number: int          # 1-based
    text: str                 # 整页纯文本
    blocks: list[dict]        # 文本块列表，每块含 bbox / text / block_type
    images: list[dict]        # 图片信息列表（bbox / xref）
    width: float
    height: float


@dataclass
class PDFContent:
    """整个 PDF 的解析结果"""
    source: str                         # 文件路径
    total_pages: int
    metadata: dict                      # PDF 元数据（title/author/...）
    pages: list[PageContent] = field(default_factory=list)
    toc: list[TOCEntry] = field(default_factory=list)  # 目录 [lvl, title, page]，无则空

    @property
    def full_text(self) -> str:
        """所有页面拼接后的纯文本（页间以换行分隔）"""
        return "\n".join(p.text for p in self.pages)


class PDFExtractor:
    """
    基于 PyMuPDF（fitz）的 PDF 文本解析工具类。

    主要功能：
    - 提取全文 / 指定页范围文本
    - 提取文本块（含坐标），便于后续 chunking
    - 提取 PDF 元数据
    - 逐页迭代，适合大文件流式处理

    示例::

        extractor = PDFExtractor("path/to/book.pdf")
        content = extractor.extract()           # 解析全部页面
        print(content.metadata)
        print(content.full_text[:500])

        # 仅解析第 10-20 页（1-based，包含两端）
        content = extractor.extract(start_page=10, end_page=20)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"PDF 文件不存在：{self.path}")
        if self.path.suffix.lower() != ".pdf":
            raise ValueError(f"不支持的文件格式（需要 .pdf）：{self.path.suffix}")

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def extract(
        self,
        start_page: int = 1,
        end_page: int | None = None,
        extract_images: bool = False,
    ) -> PDFContent:
        """
        解析 PDF，返回 :class:`PDFContent`。

        :param start_page: 起始页码（1-based，含）
        :param end_page:   结束页码（1-based，含）；None 表示到末页
        :param extract_images: 是否收集图片位置信息（不解码图片数据）
        """
        with fitz.open(str(self.path)) as doc:
            total = doc.page_count
            start_idx, end_idx = self._resolve_range(start_page, end_page, total)
            metadata = self._extract_metadata(doc)
            toc = self._extract_toc(doc)

            pages: list[PageContent] = []
            for idx in range(start_idx, end_idx):
                page = doc.load_page(idx)
                pages.append(self._parse_page(page, extract_images=extract_images))

        logger.debug(
            "PDFExtractor: 解析完成 %s  共 %d 页（解析 %d 页），TOC 条目 %d",
            self.path.name, total, len(pages), len(toc),
        )
        return PDFContent(
            source=str(self.path),
            total_pages=total,
            metadata=metadata,
            pages=pages,
            toc=toc,
        )

    def iter_pages(
        self,
        start_page: int = 1,
        end_page: int | None = None,
        extract_images: bool = False,
    ) -> Iterator[PageContent]:
        """
        逐页 yield :class:`PageContent`，适合大文件流式处理。
        参数含义同 :meth:`extract`。
        """
        with fitz.open(str(self.path)) as doc:
            total = doc.page_count
            start_idx, end_idx = self._resolve_range(start_page, end_page, total)
            for idx in range(start_idx, end_idx):
                page = doc.load_page(idx)
                yield self._parse_page(page, extract_images=extract_images)

    def extract_text_only(
        self,
        start_page: int = 1,
        end_page: int | None = None,
    ) -> str:
        """仅返回拼接后的纯文本字符串，跳过块/图片信息，开销最小。"""
        parts: list[str] = []
        with fitz.open(str(self.path)) as doc:
            total = doc.page_count
            start_idx, end_idx = self._resolve_range(start_page, end_page, total)
            for idx in range(start_idx, end_idx):
                page = doc.load_page(idx)
                parts.append(page.get_text("text"))
        return "\n".join(parts)

    def get_metadata(self) -> dict:
        """仅读取 PDF 元数据，不解析正文。"""
        with fitz.open(str(self.path)) as doc:
            return self._extract_metadata(doc)

    @property
    def page_count(self) -> int:
        """返回总页数。"""
        with fitz.open(str(self.path)) as doc:
            return doc.page_count

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_range(
        start_page: int,
        end_page: int | None,
        total: int,
    ) -> tuple[int, int]:
        """将 1-based [start_page, end_page] 转换为 0-based [start_idx, end_idx)。"""
        start_idx = max(0, start_page - 1)
        end_idx = total if end_page is None else min(end_page, total)
        if start_idx >= end_idx:
            raise ValueError(
                f"页码范围无效：start_page={start_page}, end_page={end_page}, total={total}"
            )
        return start_idx, end_idx

    @staticmethod
    def _extract_toc(doc: fitz.Document) -> list[TOCEntry]:
        """提取 PDF 目录（书签/大纲），无则返回空列表。条目为 (层级, 标题, 页码 1-based)。"""
        try:
            raw = doc.get_toc(simple=True)  # list of [lvl, title, page]
        except Exception:
            return []
        out: list[TOCEntry] = []
        for item in raw:
            if len(item) >= 3:
                lvl, title, page = item[0], item[1], item[2]
                if isinstance(lvl, (int, float)) and isinstance(page, (int, float)):
                    out.append((int(lvl), str(title).strip(), int(page)))
        return out

    @staticmethod
    def _extract_metadata(doc: fitz.Document) -> dict:
        raw = doc.metadata or {}
        return {
            "title": raw.get("title", ""),
            "author": raw.get("author", ""),
            "subject": raw.get("subject", ""),
            "keywords": raw.get("keywords", ""),
            "creator": raw.get("creator", ""),
            "producer": raw.get("producer", ""),
            "creation_date": raw.get("creationDate", ""),
            "mod_date": raw.get("modDate", ""),
            "page_count": doc.page_count,
        }

    @staticmethod
    def _parse_page(page: fitz.Page, *, extract_images: bool) -> PageContent:
        rect = page.rect
        page_width = rect.width
        page_height = rect.height

        # 使用 get_text("dict") 获取带字号、粗体等信息的块，便于识别标题
        raw_dict = page.get_text("dict")
        dict_blocks = raw_dict.get("blocks") or []

        # 收集本页所有 span 的字号，用于估计“正文”字号（取中位数）
        all_sizes: list[float] = []
        for blk in dict_blocks:
            for line in blk.get("lines") or []:
                for span in line.get("spans") or []:
                    s = span.get("size")
                    if s is not None and s > 0:
                        all_sizes.append(float(s))
        body_font_size = float(
            sorted(all_sizes)[len(all_sizes) // 2]
        ) if all_sizes else 12.0

        # 传统 blocks 含文本+图片，与 dict 的文本块顺序一致（dict 无图片块）
        raw_blocks = page.get_text("blocks")
        blocks: list[dict] = []
        text_parts: list[str] = []
        dict_idx = 0

        for block_no, raw_blk in enumerate(raw_blocks):
            x0, y0, x1, y1, content, _, block_type = raw_blk
            text = content.strip()

            is_heading = False
            heading_text = ""
            if block_type == 0 and dict_idx < len(dict_blocks):
                dict_blk = dict_blocks[dict_idx]
                dict_idx += 1
                if dict_blk.get("lines"):
                    is_heading, heading_text = PDFExtractor._is_heading_block(
                        dict_blk,
                        body_font_size,
                        page_width,
                        page_height,
                        blocks,
                    )

            blocks.append({
                "bbox": (x0, y0, x1, y1),
                "text": text,
                "block_no": block_no,
                "block_type": block_type,
                "is_heading": is_heading,
                "heading_text": heading_text if is_heading else "",
            })
            if text:
                text_parts.append(text)

        # 纯文本与块顺序一致，便于后续按块偏移推算 section
        text = "\n".join(text_parts)

        # 图片位置（可选）：dict 里无图片块，从 raw_blocks 补
        images: list[dict] = []
        if extract_images:
            for img in page.get_images(full=True):
                xref = img[0]
                bbox_list = page.get_image_rects(xref)
                for bbox in bbox_list:
                    images.append({"xref": xref, "bbox": tuple(bbox)})

        return PageContent(
            page_number=page.number + 1,  # 转回 1-based
            text=text,
            blocks=blocks,
            images=images,
            width=rect.width,
            height=rect.height,
        )

    @staticmethod
    def _is_heading_block(
        dict_blk: dict,
        body_font_size: float,
        page_width: float,
        page_height: float,
        prev_blocks: list[dict],
    ) -> tuple[bool, str]:
        """
        根据特征判断是否为章节标题，并返回 (是否标题, 标题文本)。

        特征：大字号（比正文大 20%~30%）、粗体、居中或段前段后空白多、段落短行数少。
        """
        lines = dict_blk.get("lines") or []
        if not lines:
            return False, ""

        all_text_parts: list[str] = []
        max_size = 0.0
        any_bold = False
        line_count = 0

        for line in lines:
            line_count += 1
            for span in line.get("spans") or []:
                s = span.get("size")
                if s is not None and s > 0:
                    max_size = max(max_size, float(s))
                if span.get("flags") is not None:
                    if int(span["flags"]) & _FONT_FLAG_BOLD:
                        any_bold = True
                all_text_parts.append(span.get("text") or "")
        text = "".join(all_text_parts).strip()
        char_count = len(text)

        # 段落短、行数少
        short_paragraph = (
            char_count <= _HEADING_MAX_CHARS and line_count <= _HEADING_MAX_LINES
        )
        if not short_paragraph:
            return False, ""

        # 大字号：比正文大 20% 以上，且不要过大（避免封面大标题）
        size_ratio = max_size / body_font_size if body_font_size > 0 else 0
        large_size = (
            _HEADING_SIZE_RATIO_MIN <= size_ratio <= _HEADING_SIZE_RATIO_MAX
        )

        # 粗体
        bold_ok = any_bold

        # 居中：块水平中心接近页面中心
        bbox = dict_blk.get("bbox")
        if bbox and len(bbox) >= 4:
            block_cx = (bbox[0] + bbox[2]) / 2
            page_cx = page_width / 2
            centered = abs(block_cx - page_cx) <= page_width * _HEADING_CENTER_TOLERANCE
        else:
            centered = False

        # 段前段后空白多：与上一块的垂直间距
        gap_above = 0.0
        if bbox and prev_blocks:
            last = prev_blocks[-1]
            last_bbox = last.get("bbox")
            if last_bbox and len(last_bbox) >= 4 and len(bbox) >= 4:
                gap_above = bbox[1] - last_bbox[3]
        extra_vertical = gap_above >= _HEADING_VERTICAL_GAP_MIN

        # 满足：大字号 + （粗体 或 居中 或 段前段后空白多）
        is_heading = large_size and (bold_ok or centered or extra_vertical)
        return is_heading, text if is_heading else ""
