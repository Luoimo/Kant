# _*_ coding:utf-8 _*_
# @Time:2026/3/10
# @Author:Chloe
# @File:text_cleaner.py
# @Project:Kant

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
    clean_dashes,
    clean_extra_whitespace,
    clean_ordered_bullets,
    clean_trailing_punctuation,
    group_broken_paragraphs,
    remove_punctuation,
    replace_unicode_quotes,
)

from backend.rag.extracter.pdf_extractor import (
    PDFContent,
    PageContent,
    build_page_section_map,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class CleanConfig:
    """
    TextCleaner 的清洗行为开关，所有选项默认启用最常用的安全策略。

    unstructured 原生选项
    ----------------------
    extra_whitespace    : 合并多余空白符
    bullets             : 移除无序列表符号（•、◦ 等）
    dashes              : 统一破折号表示
    trailing_punctuation: 删除行尾多余标点
    ordered_bullets     : 移除有序列表编号（"1." "a)" 等）
    unicode_quotes      : 将花引号替换为直引号

    自定义选项
    ----------
    fix_hyphenation     : 修复跨行连字符断词（word-\nword → wordword）
    remove_page_numbers : 删除孤立的页码行（纯数字行）
    remove_headers_footers: 基于启发式规则删除页眉页脚
    header_footer_lines : 启发式判断页眉/页脚的最大行数（从页面首尾算）
    normalize_unicode   : NFC 规范化 Unicode 字符
    min_block_chars     : 文本块最少字符数，低于此值视为噪声块并丢弃
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
    remove_headers_footers: bool = True
    header_footer_lines: int = 1
    normalize_unicode: bool = True
    min_block_chars: int = 3


# ---------------------------------------------------------------------------
# 输出数据类
# ---------------------------------------------------------------------------

@dataclass
class CleanedPage:
    """单页清洗结果，携带原始页码信息及书本章节（来自 PDF 目录）。"""
    page_number: int
    text: str
    blocks: list[dict]          # 清洗后的文本块（已过滤噪声块）
    width: float
    height: float
    source_page: PageContent    # 原始 PageContent 引用
    chapter_title: str = ""     # 当前页所属章标题（TOC 层级 1）
    section_title: str = ""     # 当前页所属节标题（TOC 最细层级）


@dataclass
class CleanedContent:
    """整个 PDF 的清洗结果。"""
    source: str
    total_pages: int
    metadata: dict
    pages: list[CleanedPage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """所有页面清洗后文本，以双换行分段。"""
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class TextCleaner:
    """
    基于 ``unstructured`` 的 PDF 文本清洗工具类。

    接受 :class:`~backend.rag.extracter.pdf_extractor.PDFContent` 或
    :class:`~backend.rag.extracter.pdf_extractor.PageContent` 作为输入，
    返回对应的 :class:`CleanedContent` / :class:`CleanedPage`，
    也可直接清洗裸字符串。

    示例::

        from backend.rag.extracter import PDFExtractor
        from backend.rag.cleaner import TextCleaner

        content = PDFExtractor("book.pdf").extract()
        cleaner = TextCleaner()                        # 默认配置
        cleaned = cleaner.clean_content(content)
        print(cleaned.full_text[:500])

        # 自定义：不删除页眉页脚，保留有序列表编号
        cfg = CleanConfig(remove_headers_footers=False, ordered_bullets=True)
        cleaner = TextCleaner(cfg)
    """

    def __init__(self, config: CleanConfig | None = None) -> None:
        self.config = config or CleanConfig()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def clean_content(self, content: PDFContent) -> CleanedContent:
        """清洗整个 :class:`PDFContent`，返回 :class:`CleanedContent`。"""
        page_section_map = build_page_section_map(
            content.toc, content.total_pages
        )
        cleaned_pages: list[CleanedPage] = []
        for p in content.pages:
            cp = self.clean_page(p)
            chapter, section = page_section_map.get(
                p.page_number, ("", "")
            )
            cleaned_pages.append(
                CleanedPage(
                    page_number=cp.page_number,
                    text=cp.text,
                    blocks=cp.blocks,
                    width=cp.width,
                    height=cp.height,
                    source_page=cp.source_page,
                    chapter_title=chapter,
                    section_title=section,
                )
            )
        logger.debug(
            "TextCleaner: 完成 %s，共 %d 页",
            content.source, len(cleaned_pages),
        )
        return CleanedContent(
            source=content.source,
            total_pages=content.total_pages,
            metadata=copy.copy(content.metadata),
            pages=cleaned_pages,
        )

    def clean_page(self, page: PageContent) -> CleanedPage:
        """清洗单个 :class:`PageContent`，返回 :class:`CleanedPage`。"""
        cfg = self.config

        # 1. 清洗文本块，过滤噪声（会保留 extractor 写入的 is_heading / heading_text）
        cleaned_blocks = self._clean_blocks(page.blocks, page.height)

        # 2. 对每块单独做管道清洗，再拼接为页面文本，便于为每块记录 start_offset/end_offset（供 chunker 推算 section_title）
        for blk in cleaned_blocks:
            blk["text"] = self._apply_pipeline(blk["text"])

        # 3. 计算每块在页面文本中的起止偏移
        cur = 0
        for blk in cleaned_blocks:
            t = (blk.get("text") or "").strip()
            blk["start_offset"] = cur
            if t:
                cur += len(t) + 1  # 文本 + 换行
            blk["end_offset"] = cur

        block_texts = [b["text"].strip() for b in cleaned_blocks if b["text"].strip()]
        text = "\n".join(block_texts) if block_texts else ""

        return CleanedPage(
            page_number=page.page_number,
            text=text,
            blocks=cleaned_blocks,
            width=page.width,
            height=page.height,
            source_page=page,
            chapter_title="",
            section_title="",
        )

    def clean_text(self, text: str) -> str:
        """直接清洗裸字符串，返回清洗后的字符串。"""
        return self._apply_pipeline(text)

    def clean_pages(self, pages: Sequence[PageContent]) -> list[CleanedPage]:
        """批量清洗页面列表。"""
        return [self.clean_page(p) for p in pages]

    # ------------------------------------------------------------------
    # 内部管道
    # ------------------------------------------------------------------

    def _apply_pipeline(self, text: str) -> str:
        """按顺序应用全部已启用的清洗步骤。"""
        cfg = self.config

        if cfg.normalize_unicode:
            text = self._normalize_unicode(text)

        # 将制表符统一替换为空格（unstructured clean_extra_whitespace 不处理 \t）
        if cfg.extra_whitespace:
            text = re.sub(r"\t", " ", text)

        if cfg.fix_hyphenation:
            text = self._fix_hyphenation(text)

        if cfg.remove_page_numbers:
            text = self._remove_page_numbers(text)

        if cfg.unicode_quotes:
            # 显式替换最常见的 Unicode 花引号，兼容各版本 unstructured
            text = (
                text
                .replace("\u201c", '"').replace("\u201d", '"')   # " "
                .replace("\u2018", "'").replace("\u2019", "'")   # ' '
                .replace("\u201a", "'").replace("\u201b", "'")   # ‚ ‛
            )
            text = replace_unicode_quotes(text)  # unstructured 覆盖其他罕见花引号

        # unstructured.clean() 聚合多个选项
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

        # 修复 PDF 常见的段落断行问题
        text = group_broken_paragraphs(text)

        # 最终再压一次空白
        if cfg.extra_whitespace:
            text = clean_extra_whitespace(text)

        return text.strip()

    def _clean_blocks(self, blocks: list[dict], page_height: float) -> list[dict]:
        """
        过滤并清洗文本块列表：
        - 丢弃图片块（block_type == 1）
        - 丢弃字符数低于 min_block_chars 的噪声块
        - 根据 bbox 纵坐标启发式删除页眉/页脚块
        - 对每个块的 text 执行轻量清洗
        """
        cfg = self.config
        result: list[dict] = []

        for blk in blocks:
            if blk.get("block_type") == 1:  # 图片块，跳过
                continue

            txt = blk.get("text", "").strip()

            if len(txt) < cfg.min_block_chars:
                continue

            if cfg.remove_page_numbers and self._is_page_number(txt):
                continue

            if cfg.remove_headers_footers and self._is_header_footer(blk, page_height):
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

    # 匹配孤立页码：整行仅含数字（可含空格、连字符、"第X页"等格式）
    _PAGE_NUMBER_RE = re.compile(
        r"""
        ^[\s\-–—]*          # 可选前缀空白/破折号
        (
            第\s*\d+\s*页   # 中文页码
          | \d+             # 纯数字
          | [ivxlcdmIVXLCDM]+  # 罗马数字
        )
        [\s\-–—.]*$         # 可选后缀
        """,
        re.VERBOSE,
    )

    # 修复连字符断词：word-\nword → wordword，word-\n word → word word
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

    def _is_header_footer(self, blk: dict, page_height: float) -> bool:
        """
        启发式判断：文本块 bbox 的顶边或底边距页面边缘
        在 header_footer_lines 行高（约 20px/行）以内视为页眉/页脚。
        """
        if not page_height:
            return False
        threshold = self.config.header_footer_lines * 20.0
        y0, y1 = blk["bbox"][1], blk["bbox"][3]
        return y0 < threshold or y1 > (page_height - threshold)
