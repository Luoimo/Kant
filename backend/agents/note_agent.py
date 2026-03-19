from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import sys

from langchain_core.documents import Document

from backend.config import get_settings
from backend.llm.openai_client import get_llm
from backend.rag.chroma.chroma_store import ChromaStore
from backend.xai.citation import Citation, build_citations

sep = "\n\n"

NOTE_SYSTEM_PROMPT = """你是"笔记整理助手（NoteAgent）"，擅长将书籍内容或零散文字整理成结构化笔记。

核心职责：
1. 将检索到的内容或用户提供的文字，整理成清晰的 Markdown 格式笔记。
2. 笔记应包含：分级标题、要点列表、关键概念、页码引用（如有）、待探索问题（可选）。
3. 重点在于"归纳与结构化"，而非一问一答。
4. 如果内容不足，如实说明，不要编造书中不存在的内容。

输出格式：
- 使用 Markdown 标题（## ### ####）组织层次
- 要点用 - 列表
- 关键概念可加粗 **概念**
- 页码引用格式：（p.XX）或（见某书，p.XX）
"""


@dataclass(frozen=True)
class NoteResult:
    answer: str
    citations: list[Citation]
    retrieved_docs: list[Document]


class NoteAgent:
    def __init__(
        self,
        *,
        store: ChromaStore | None = None,
        llm=None,
        k: int = 8,
    ) -> None:
        if store is None:
            settings = get_settings()
            store = ChromaStore(collection_name=settings.chroma_database)

        self.store = store
        self.llm = llm or get_llm(temperature=0.3)
        self.k = k

    def run(
        self,
        *,
        query: str,
        book_source: str | None = None,
        raw_text: str | None = None,
    ) -> NoteResult:
        # 路径 1：有 RAG（book_source 指定时）
        if book_source or (not raw_text):
            filter_ = {"source": book_source} if book_source else None
            docs = self.store.similarity_search(query, k=self.k, filter=filter_)
            citations = build_citations(docs)

            print(
                f"[NoteAgent] query={query!r}, "
                f"book_source={book_source!r}, hits={len(docs)}",
                file=sys.stdout,
            )

            if not docs and not raw_text:
                return NoteResult(
                    answer="本地书库没有检索到相关内容，无法生成笔记。请先将相关 PDF 入库，或提供待整理的原文。",
                    citations=[],
                    retrieved_docs=[],
                )

            answer = self._synthesize_notes(query, docs, raw_text)
            return NoteResult(answer=answer, citations=citations, retrieved_docs=docs)

        # 路径 2：纯文本整理（raw_text，无 RAG）
        print(f"[NoteAgent] raw_text mode, len={len(raw_text)}", file=sys.stdout)
        answer = self._synthesize_notes(query, [], raw_text)
        return NoteResult(answer=answer, citations=[], retrieved_docs=[])

    def _synthesize_notes(
        self,
        query: str,
        docs: list[Document],
        raw_text: str | None = None,
    ) -> str:
        content_blocks: list[str] = []

        for i, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            title = meta.get("book_title") or "未知书名"
            pages = meta.get("section_indices") or ""
            content_blocks.append(
                f"[片段{i}] 书名：{title}  页码：{pages}\n"
                f"{(d.page_content or '').strip()}"
            )

        if raw_text:
            content_blocks.append(f"[用户提供文本]\n{raw_text.strip()}")

        user_prompt = f"""用户笔记整理请求：
{query}

请将以下内容整理成结构化 Markdown 笔记：

{'【书库检索片段 + 用户文本】：' if docs and raw_text else '【书库检索片段】：' if docs else '【用户提供文本】：'}
{sep.join(content_blocks)}

整理要求：
1. 使用 Markdown 格式：分级标题、要点列表、关键概念加粗。
2. 对重要内容尽量标注页码引用（例如：（p.XX）或（见某书，p.XX））。
3. 可在末尾添加"待探索问题"小节（可选）。
4. 只整理实际出现的内容，不要添加书中没有的信息。"""

        msg = self.llm.invoke(
            [
                {"role": "system", "content": NOTE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        return getattr(msg, "content", str(msg))


def notes_node(state: dict[str, Any], *, agent: NoteAgent) -> dict[str, Any]:
    """节点函数：读取 notes_query / notes_book_source / notes_raw_text，写回 answer/citations/retrieved_docs_count。"""
    query: str = state.get("notes_query", "") or state.get("user_input", "")
    book_source: str | None = state.get("notes_book_source") or state.get("book_source")
    raw_text: str | None = state.get("notes_raw_text")

    result = agent.run(query=query, book_source=book_source, raw_text=raw_text)
    return {
        "answer": result.answer,
        "citations": result.citations,
        "retrieved_docs_count": len(result.retrieved_docs),
    }


__all__ = ["NoteAgent", "NoteResult", "notes_node"]
