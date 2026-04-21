# _*_ coding:utf-8 _*_
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Literal, Sequence

from rag.cleaner.text_cleaner import CleanedBookContent, CleanedSection

logger = logging.getLogger(__name__)

# 句末标点集合，用于"软切分"时从硬切点向右寻找自然边界
_END_PUNCTS = "。！？；.!?;"


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class ChunkConfig:
    """
    TextChunker 行为配置。

    splitter        : "recursive"（按字符递归拆分，默认）| "token"（按 token 计数拆分）
    chunk_size      : recursive 模式下单块最大字符数；token 模式下为最大 token 数
    chunk_overlap   : 相邻块重叠量（字符或 token，单位与 chunk_size 一致）
    encoding_name   : token 模式使用的 tiktoken 编码，默认 cl100k_base（GPT-4/3.5）
    separators      : recursive 模式下的分隔符优先级列表（中英文混合）
    section_aware   : True → 逐章节独立切分，精确保留章节号；
                      False → 全文拼接后统一切分，跨章节 chunk 携带多章节号
    min_chunk_chars : 切分后最短保留字符数，低于此值的碎片直接丢弃
    """
    splitter: Literal["recursive", "token"] = "recursive"
    chunk_size: int = 1024
    chunk_overlap: int = 64
    encoding_name: str = "cl100k_base"
    separators: list[str] = field(default_factory=lambda: [
        "\n\n", "\n", "。", "！", "？", "；",
        ".", "!", "?", ";", " ", "",
    ])
    section_aware: bool = True
    min_chunk_chars: int = 20


# ---------------------------------------------------------------------------
# 输出数据类
# ---------------------------------------------------------------------------

@dataclass
class ChunkMeta:
    """随每个 TextChunk 携带的位置、来源与书本章节信息。"""
    source: str
    section_indices: list[int]   # 该 chunk 涉及的章节索引（0-based）
    chunk_index: int             # 在整篇文档中的全局顺序（0-based）
    book_title: str = ""
    author: str = ""
    chapter_title: str = ""      # 所属章标题（来自 TOC）
    section_title: str = ""      # 所属节标题（来自 TOC）


@dataclass
class TextChunk:
    """切分结果单元。"""
    chunk_id: str       # text 内容的 SHA-256 前 16 位，便于去重比较
    text: str
    char_count: int
    metadata: ChunkMeta

    def to_dict(self) -> dict:
        """转换为可序列化字典，方便后续写入向量库。"""
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


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class TextChunker:
    """
    基于软切分的 EPUB 文本切分工具类。

    接受 CleanedBookContent 或 CleanedSection 作为输入，
    输出带完整元数据的 TextChunk 列表，可直接送入向量化流程。

    示例::

        content = EpubExtractor("book.epub").extract()
        cleaned = TextCleaner().clean_content(content)
        chunks = TextChunker().chunk_content(cleaned)
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def chunk_content(self, content: CleanedBookContent) -> list[TextChunk]:
        """切分整个 CleanedBookContent，返回带全局 chunk_index 的 TextChunk 列表。"""
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

    def chunk_section(
        self,
        section: CleanedSection,
        *,
        source: str = "",
        book_metadata: dict | None = None,
        index_offset: int = 0,
    ) -> list[TextChunk]:
        """切分单个 CleanedSection。"""
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

    def chunk_text(
        self,
        text: str,
        *,
        source: str = "",
        section_indices: list[int] | None = None,
        book_metadata: dict | None = None,
        index_offset: int = 0,
    ) -> list[TextChunk]:
        """直接切分裸字符串，适用于单元测试或临时场景。"""
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

    # ------------------------------------------------------------------
    # 内部：两种切分模式
    # ------------------------------------------------------------------

    def _chunk_section_aware(self, content: CleanedBookContent) -> list[TextChunk]:
        """逐章节切分，精确保留章节号。"""
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

    def _chunk_fulltext(self, content: CleanedBookContent) -> list[TextChunk]:
        """全文拼接后统一切分，跨章节 chunk 携带多章节号。"""
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

    # ------------------------------------------------------------------
    # 内部：构建 TextChunk
    # ------------------------------------------------------------------

    def _build_chunks(
        self,
        raw_chunks: list[str],
        *,
        section_indices: list[int],
        source: str,
        book_title: str,
        author: str,
        index_offset: int = 0,
        chapter_title: str = "",
        section_title: str = "",
    ) -> list[TextChunk]:
        result: list[TextChunk] = []
        for i, text in enumerate(raw_chunks):
            text = text.strip()
            if len(text) < self.config.min_chunk_chars:
                continue
            result.append(TextChunk(
                chunk_id=_sha256_id(text),
                text=text,
                char_count=len(text),
                metadata=ChunkMeta(
                    source=source,
                    section_indices=section_indices,
                    chunk_index=index_offset + len(result),
                    book_title=book_title,
                    author=author,
                    chapter_title=chapter_title,
                    section_title=section_title,
                ),
            ))
        return result

    def _section_titles_for_chunks(
        self,
        raw_chunks: list[str],
        page_text: str,
        blocks: list[dict],
    ) -> list[str]:
        """
        根据每个 chunk 在 page_text 中的起始位置，取「不晚于该位置」的最后一个
        is_heading 块的 heading_text 作为该 chunk 的 section_title。
        """
        if not raw_chunks or not page_text:
            return [""] * len(raw_chunks)

        chunk_starts: list[int] = []
        search_from = 0
        for chunk in raw_chunks:
            prefix = (chunk.strip()[:50] or chunk.strip()).strip()
            if not prefix:
                chunk_starts.append(0)
                continue
            idx = page_text.find(prefix, search_from)
            if idx == -1:
                idx = page_text.find(prefix)
            chunk_starts.append(max(0, idx if idx != -1 else 0))
            if idx >= 0:
                search_from = idx + len(prefix)

        result: list[str] = []
        for chunk_start in chunk_starts:
            section_title = ""
            for blk in blocks:
                if not blk.get("is_heading"):
                    continue
                start_off = blk.get("start_offset", 0)
                if start_off <= chunk_start:
                    section_title = blk.get("heading_text") or ""
            result.append(section_title)
        return result

    def _build_chunks_with_section_titles(
        self,
        raw_chunks: list[str],
        *,
        section_indices: list[int],
        source: str,
        book_title: str,
        author: str,
        index_offset: int = 0,
        chapter_title: str = "",
        section_titles: list[str],
    ) -> list[TextChunk]:
        """与 _build_chunks 类似，但每个 chunk 使用对应的 section_titles[i]。"""
        result: list[TextChunk] = []
        for i, text in enumerate(raw_chunks):
            text = text.strip()
            if len(text) < self.config.min_chunk_chars:
                continue
            section_title = (
                section_titles[i] if i < len(section_titles) else ""
            )
            result.append(TextChunk(
                chunk_id=_sha256_id(text),
                text=text,
                char_count=len(text),
                metadata=ChunkMeta(
                    source=source,
                    section_indices=section_indices,
                    chunk_index=index_offset + len(result),
                    book_title=book_title,
                    author=author,
                    chapter_title=chapter_title,
                    section_title=section_title,
                ),
            ))
        return result

    # ------------------------------------------------------------------
    # 内部：软切分
    # ------------------------------------------------------------------

    def _soft_split(
        self,
        text: str,
        *,
        max_len: int,
        overlap: int,
        max_shift: int | None = None,
    ) -> list[str]:
        text = text or ""
        n = len(text)
        if n == 0:
            return []

        if max_shift is None:
            max_shift = max_len // 4
            if max_shift <= 0:
                max_shift = 32

        chunks: list[str] = []
        pos = 0

        while pos < n:
            hard_end = min(pos + max_len, n)
            if hard_end >= n:
                tail = text[pos:].strip()
                if len(tail) >= self.config.min_chunk_chars:
                    chunks.append(tail)
                break

            search_end = min(hard_end + max_shift, n)
            cut_idx: int | None = None

            for i in range(hard_end, search_end):
                if text[i] in _END_PUNCTS:
                    cut_idx = i + 1
                    break

            end = cut_idx if cut_idx is not None else hard_end
            chunk_text = text[pos:end].strip()

            if len(chunk_text) >= self.config.min_chunk_chars:
                chunks.append(chunk_text)

            if end >= n:
                break

            next_pos = max(0, end - overlap)
            if next_pos <= pos:
                next_pos = end
            pos = next_pos

        return chunks

    def _soft_split_positions(
        self,
        text: str,
        *,
        max_len: int,
        overlap: int,
        max_shift: int | None = None,
    ) -> list[tuple[int, int]]:
        text = text or ""
        n = len(text)
        if n == 0:
            return []

        if max_shift is None:
            max_shift = max_len // 4
            if max_shift <= 0:
                max_shift = 32

        positions: list[tuple[int, int]] = []
        pos = 0

        while pos < n:
            hard_end = min(pos + max_len, n)
            if hard_end >= n:
                end = n
                if end > pos:
                    positions.append((pos, end))
                break

            search_end = min(hard_end + max_shift, n)
            cut_idx: int | None = None

            for i in range(hard_end, search_end):
                if text[i] in _END_PUNCTS:
                    cut_idx = i + 1
                    break

            end = cut_idx if cut_idx is not None else hard_end

            if end > pos:
                positions.append((pos, end))

            if end >= n:
                break

            next_pos = max(0, end - overlap)
            if next_pos <= pos:
                next_pos = end
            pos = next_pos

        return positions

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _sha256_id(text: str) -> str:
    """取文本 SHA-256 的前 16 位十六进制作为轻量 ID。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
