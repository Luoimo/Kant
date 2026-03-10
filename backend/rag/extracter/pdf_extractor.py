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

            pages: list[PageContent] = []
            for idx in range(start_idx, end_idx):
                page = doc.load_page(idx)
                pages.append(self._parse_page(page, extract_images=extract_images))

        logger.debug(
            "PDFExtractor: 解析完成 %s  共 %d 页（解析 %d 页）",
            self.path.name, total, len(pages),
        )
        return PDFContent(
            source=str(self.path),
            total_pages=total,
            metadata=metadata,
            pages=pages,
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

        # 文本块：每个 block = (x0, y0, x1, y1, text, block_no, block_type)
        raw_blocks = page.get_text("blocks")
        blocks: list[dict] = []
        for b in raw_blocks:
            x0, y0, x1, y1, content, block_no, block_type = b
            blocks.append({
                "bbox": (x0, y0, x1, y1),
                "text": content.strip(),
                "block_no": block_no,
                "block_type": block_type,  # 0=text, 1=image
            })

        # 纯文本（保留段落换行）
        text = page.get_text("text")

        # 图片位置（可选）
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
