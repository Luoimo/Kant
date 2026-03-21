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
