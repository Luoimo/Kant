# _*_ coding:utf-8 _*_
# @Time:2026/3/10
# @Author:Chloe
# @File:text_chunker.py
# @Project:Kant

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Literal, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

from backend.rag.cleaner.text_cleaner import CleanedContent, CleanedPage

logger = logging.getLogger(__name__)

# 全文模式下插入页边界标记，用于切分后反推页码
_PAGE_MARKER = "\n<<<PAGE:{page}>>>\n"
_PAGE_MARKER_RE = __import__("re").compile(r"<<<PAGE:(\d+)>>>")


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
    page_aware      : True → 逐页独立切分，精确保留页码；
                      False → 全文拼接后统一切分，跨页 chunk 携带多页码
    min_chunk_chars : 切分后最短保留字符数，低于此值的碎片直接丢弃
    """
    splitter: Literal["recursive", "token"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    encoding_name: str = "cl100k_base"
    separators: list[str] = field(default_factory=lambda: [
        "\n\n", "\n", "。", "！", "？", "；",
        ".", "!", "?", ";", " ", "",
    ])
    page_aware: bool = True
    min_chunk_chars: int = 20


# ---------------------------------------------------------------------------
# 输出数据类
# ---------------------------------------------------------------------------

@dataclass
class ChunkMeta:
    """随每个 TextChunk 携带的位置与来源信息。"""
    source: str             # PDF 文件路径
    page_numbers: list[int] # 该 chunk 涉及的页码（1-based）
    chunk_index: int        # 在整篇文档中的全局顺序（0-based）
    pdf_title: str = ""
    pdf_author: str = ""


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
            "page_numbers": self.metadata.page_numbers,
            "chunk_index": self.metadata.chunk_index,
            "pdf_title": self.metadata.pdf_title,
            "pdf_author": self.metadata.pdf_author,
        }


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class TextChunker:
    """
    基于 LangChain TextSplitter 的文本切分工具类。

    接受 :class:`~backend.rag.cleaner.text_cleaner.CleanedContent` 或
    :class:`~backend.rag.cleaner.text_cleaner.CleanedPage` 作为输入，
    输出带完整元数据的 :class:`TextChunk` 列表，可直接送入向量化流程。

    支持两种拆分器：

    * ``recursive``（默认）：:class:`~langchain_text_splitters.RecursiveCharacterTextSplitter`，
      按段落 → 句子 → 词语优先级递归拆分，适合哲学/人文类长文本。
    * ``token``：:class:`~langchain_text_splitters.TokenTextSplitter`，
      按 tiktoken token 数拆分，精确控制送入 LLM 的上下文大小。

    两种拆分模式：

    * ``page_aware=True``（默认）：逐页独立切分，每个 chunk 的页码精确。
    * ``page_aware=False``：全文拼接后统一切分，跨页 chunk 自动收集涉及的页码列表。

    示例::

        from backend.rag.extracter import PDFExtractor
        from backend.rag.cleaner import TextCleaner
        from backend.rag.chunker import TextChunker, ChunkConfig

        content = PDFExtractor("kant.pdf").extract()
        cleaned = TextCleaner().clean_content(content)

        # 默认配置（recursive，512 字符，页面感知）
        chunks = TextChunker().chunk_content(cleaned)

        # token 模式，256 tokens，跨页拼接
        cfg = ChunkConfig(splitter="token", chunk_size=256, page_aware=False)
        chunks = TextChunker(cfg).chunk_content(cleaned)
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()
        self._splitter = self._build_splitter(self.config)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def chunk_content(self, content: CleanedContent) -> list[TextChunk]:
        """
        切分整个 :class:`CleanedContent`，返回带全局 chunk_index 的 :class:`TextChunk` 列表。
        """
        if self.config.page_aware:
            chunks = self._chunk_page_aware(content)
        else:
            chunks = self._chunk_fulltext(content)

        logger.debug(
            "TextChunker: %s → %d 个 chunk（%s 模式，page_aware=%s）",
            content.source, len(chunks),
            self.config.splitter, self.config.page_aware,
        )
        return chunks

    def chunk_page(
        self,
        page: CleanedPage,
        *,
        source: str = "",
        pdf_metadata: dict | None = None,
        index_offset: int = 0,
    ) -> list[TextChunk]:
        """
        切分单个 :class:`CleanedPage`。

        :param source:       PDF 文件路径，用于 chunk 元数据
        :param pdf_metadata: PDF 元数据字典（含 title/author 等）
        :param index_offset: chunk_index 起始偏移，多页拼接时用于保持全局编号连续
        """
        if not page.text.strip():
            return []
        meta = pdf_metadata or {}
        raw_chunks = self._splitter.split_text(page.text)
        return self._build_chunks(
            raw_chunks,
            page_numbers=[page.page_number],
            source=source,
            pdf_title=meta.get("title", ""),
            pdf_author=meta.get("author", ""),
            index_offset=index_offset,
        )

    def chunk_text(
        self,
        text: str,
        *,
        source: str = "",
        page_numbers: list[int] | None = None,
        pdf_metadata: dict | None = None,
        index_offset: int = 0,
    ) -> list[TextChunk]:
        """直接切分裸字符串，适用于单元测试或临时场景。"""
        meta = pdf_metadata or {}
        raw_chunks = self._splitter.split_text(text)
        return self._build_chunks(
            raw_chunks,
            page_numbers=page_numbers or [],
            source=source,
            pdf_title=meta.get("title", ""),
            pdf_author=meta.get("author", ""),
            index_offset=index_offset,
        )

    # ------------------------------------------------------------------
    # 内部：两种切分模式
    # ------------------------------------------------------------------

    def _chunk_page_aware(self, content: CleanedContent) -> list[TextChunk]:
        """逐页切分，精确保留页码。"""
        all_chunks: list[TextChunk] = []
        for page in content.pages:
            new_chunks = self.chunk_page(
                page,
                source=content.source,
                pdf_metadata=content.metadata,
                index_offset=len(all_chunks),
            )
            all_chunks.extend(new_chunks)
        return all_chunks

    def _chunk_fulltext(self, content: CleanedContent) -> list[TextChunk]:
        """
        全文拼接后统一切分。
        在各页边界处插入隐形标记，切分后通过扫描标记还原每个 chunk 涉及的页码。
        """
        parts: list[str] = []
        for page in content.pages:
            if page.text.strip():
                parts.append(_PAGE_MARKER.format(page=page.page_number))
                parts.append(page.text)
        full_text = "".join(parts)

        raw_chunks = self._splitter.split_text(full_text)

        # 追踪每个 raw_chunk 出现在 full_text 的位置，统计涉及页码
        chunks = self._build_chunks_with_page_inference(
            raw_chunks, full_text,
            source=content.source,
            pdf_title=content.metadata.get("title", ""),
            pdf_author=content.metadata.get("author", ""),
        )
        return chunks

    # ------------------------------------------------------------------
    # 内部：构建 TextChunk
    # ------------------------------------------------------------------

    def _build_chunks(
        self,
        raw_chunks: list[str],
        *,
        page_numbers: list[int],
        source: str,
        pdf_title: str,
        pdf_author: str,
        index_offset: int = 0,
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
                    page_numbers=page_numbers,
                    chunk_index=index_offset + len(result),
                    pdf_title=pdf_title,
                    pdf_author=pdf_author,
                ),
            ))
        return result

    def _build_chunks_with_page_inference(
        self,
        raw_chunks: list[str],
        full_text: str,
        *,
        source: str,
        pdf_title: str,
        pdf_author: str,
    ) -> list[TextChunk]:
        """
        根据 chunk 文本在 full_text 中的位置，反推该 chunk 涉及哪些页码。
        由于 overlap，使用滑动搜索而非精确位置匹配。
        """
        # 预构建 (start_pos, page_no) 列表，供快速查找
        page_positions: list[tuple[int, int]] = [
            (m.start(), int(m.group(1)))
            for m in _PAGE_MARKER_RE.finditer(full_text)
        ]

        result: list[TextChunk] = []
        search_start = 0   # 滑动窗口，避免每次从头搜索

        for raw in raw_chunks:
            clean_text = _PAGE_MARKER_RE.sub("", raw).strip()
            if len(clean_text) < self.config.min_chunk_chars:
                continue

            # 在 full_text 中定位 chunk（取首次出现的非标记内容前缀定位）
            prefix = clean_text[:30].strip()
            pos = full_text.find(prefix, search_start)
            if pos == -1:
                pos = full_text.find(prefix)   # fallback：从头找

            chunk_end = pos + len(raw) if pos != -1 else len(full_text)

            # 收集 [pos, chunk_end) 范围内涉及的页码
            pages_in_chunk: list[int] = []
            for mark_pos, page_no in page_positions:
                if pos <= mark_pos < chunk_end:
                    pages_in_chunk.append(page_no)
            # 如果 chunk 起点在某页内（标记在 chunk 之前），补充该页
            if not pages_in_chunk:
                for mark_pos, page_no in reversed(page_positions):
                    if mark_pos <= pos:
                        pages_in_chunk = [page_no]
                        break

            # 更新滑动窗口（允许 overlap 回退）
            if pos != -1:
                search_start = max(0, pos - self.config.chunk_overlap)

            result.append(TextChunk(
                chunk_id=_sha256_id(clean_text),
                text=clean_text,
                char_count=len(clean_text),
                metadata=ChunkMeta(
                    source=source,
                    page_numbers=sorted(set(pages_in_chunk)),
                    chunk_index=len(result),
                    pdf_title=pdf_title,
                    pdf_author=pdf_author,
                ),
            ))
        return result

    # ------------------------------------------------------------------
    # 工厂
    # ------------------------------------------------------------------

    @staticmethod
    def _build_splitter(
        cfg: ChunkConfig,
    ) -> RecursiveCharacterTextSplitter | TokenTextSplitter:
        if cfg.splitter == "token":
            return TokenTextSplitter(
                encoding_name=cfg.encoding_name,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
            )
        # recursive（默认）
        return RecursiveCharacterTextSplitter(
            separators=cfg.separators,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _sha256_id(text: str) -> str:
    """取文本 SHA-256 的前 16 位十六进制作为轻量 ID。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
