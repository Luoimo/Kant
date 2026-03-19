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

PLAN_SYSTEM_PROMPT = """你是"阅读计划助手（ReadingPlanAgent）"，专门帮助用户制定个性化的阅读计划。

核心职责：
1. 根据用户需求和书库中的书籍信息，制定合理的阅读计划。
2. 计划应包含：每日/每周阅读安排、章节划分、时间估算、阅读目标、进度检查点。
3. 语气友好，计划切实可行，不要过于理想化。

输出格式（Markdown）：
- ## 阅读目标
- ## 书单 / 章节安排
- ## 每日/每周计划表
- ## 阅读建议与技巧
- ## 进度检查点（可选）
"""


@dataclass(frozen=True)
class ReadingPlanResult:
    answer: str
    citations: list[Citation]
    retrieved_docs: list[Document]


class ReadingPlanAgent:
    def __init__(
        self,
        *,
        store: ChromaStore | None = None,
        llm=None,
        k: int = 6,
    ) -> None:
        if store is None:
            settings = get_settings()
            store = ChromaStore(collection_name=settings.chroma_database)

        self.store = store
        self.llm = llm or get_llm(temperature=0.4)
        self.k = k

    def run(self, *, query: str, book_source: str | None = None, memory_context: str = "") -> ReadingPlanResult:
        # 获取书库中的可用书目
        available_sources: list[str] = []
        try:
            available_sources = self.store.list_sources()
        except Exception as e:
            print(f"[ReadingPlanAgent] list_sources failed: {e}", file=sys.stdout)

        # 如果指定了书源，检索相关元数据 chunks（目录、标题等）
        docs: list[Document] = []
        if book_source:
            filter_ = {"source": book_source}
            docs = self.store.similarity_search(query, k=self.k, filter=filter_)
        elif available_sources:
            docs = self.store.similarity_search(query, k=self.k)

        citations = build_citations(docs)

        print(
            f"[ReadingPlanAgent] query={query!r}, "
            f"book_source={book_source!r}, "
            f"available_sources={len(available_sources)}, hits={len(docs)}",
            file=sys.stdout,
        )

        answer = self._generate_plan(query, docs, available_sources, book_source, memory_context=memory_context)
        return ReadingPlanResult(answer=answer, citations=citations, retrieved_docs=docs)

    def _generate_plan(
        self,
        query: str,
        docs: list[Document],
        available_sources: list[str],
        book_source: str | None,
        memory_context: str = "",
    ) -> str:
        context_parts: list[str] = []

        if available_sources:
            context_parts.append(
                "【书库中的可用书目】：\n"
                + "\n".join(f"- {s}" for s in available_sources[:20])
            )

        for i, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            title = meta.get("book_title") or "未知书名"
            pages = meta.get("section_indices") or ""
            context_parts.append(
                f"[参考片段{i}] 书名：{title}  页码：{pages}\n"
                f"{(d.page_content or '').strip()}"
            )

        context_str = (
            sep.join(context_parts)
            if context_parts
            else "（书库暂无可用书目，请根据用户需求制定通用阅读计划）"
        )

        user_prompt = f"""用户阅读计划请求：
{query}

{f'指定书目：{book_source}' if book_source else ''}

可供参考的书库信息：
{context_str}

请根据以上信息，制定一份切实可行的 Markdown 格式阅读计划。
计划要包含：阅读目标、书单/章节安排、每日或每周时间表、阅读建议。
时间估算应合理，不要过于乐观。"""

        system = PLAN_SYSTEM_PROMPT
        if memory_context:
            system += "\n\n【用户历史阅读记录（仅供参考）】\n" + memory_context

        msg = self.llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]
        )
        return getattr(msg, "content", str(msg))


def plan_node(state: dict[str, Any], *, agent: ReadingPlanAgent) -> dict[str, Any]:
    """节点函数：读取 plan_query / plan_book_source，写回 answer/citations/retrieved_docs_count。"""
    query: str = state.get("plan_query", "") or state.get("user_input", "")
    book_source: str | None = state.get("plan_book_source") or state.get("book_source")

    memory_context: str = state.get("memory_context", "") or ""
    result = agent.run(query=query, book_source=book_source, memory_context=memory_context)
    return {
        "answer": result.answer,
        "citations": result.citations,
        "retrieved_docs_count": len(result.retrieved_docs),
    }


__all__ = ["ReadingPlanAgent", "ReadingPlanResult", "plan_node"]
