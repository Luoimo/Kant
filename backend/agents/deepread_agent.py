from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)
from typing import AsyncGenerator

from langchain_core.documents import Document
from langchain_core.tools import tool

from backend.llm.openai_client import get_llm
from backend.rag.chroma.chroma_store import ChromaStore
from backend.rag.retriever import HybridConfig, HybridRetriever
from backend.xai.citation import Citation, build_citations

sep = "\n\n"


@dataclass(frozen=True)
class DeepReadResult:
    answer: str
    citations: list[Citation]
    retrieved_docs: list[Document]


@dataclass
class DeepReadConfig:
    k: int = 6
    fetch_k: int = 20
    max_evidence: int = 8
    hybrid: HybridConfig | None = None


_SYSTEM_PROMPT = """\
你是一位治学严谨、善用苏格拉底式对话的哲学教授。学生正在阅读哲学与社科书籍，向你请教。

你有一个工具 search_books，可以检索学生正在读的书（及书库中其他书）的原文片段。

何时调用 search_books：
- 需要引用原文作为论据时
- 回答涉及书中具体论点、概念、章节内容时
- 跨书比较时，使用 scope="all_books"
- 聚焦某章节时，传入 chapter 参数

何时不需要调用工具：
- 学生表达了自己的判断，你的首要任务是追问，而不是急于检索验证
- 纯粹的哲学背景常识问题，书库原文无法提供额外价值时

对话原则：
- 学生说"我觉得/我认为"时：不急于评价对错，先问"你这个判断的依据是什么？"或从书中找一个让他进一步思考的反例/追问
- 学生说"对吗/是吗"时：不直接给是非答案，反问他的理由，再用书中证据引导他自己得出结论
- 学生要求测验时：从书中抽取一个有争议的核心命题，出一道开放题，等待他回答后再评价
- 一般内容问题：引用原文，给出有层次的分析（概念→书中论证→思想史背景）
- 书库中没有的书，用通识知识补充，末尾注明「（来自通识知识，非书库原文）」
"""


def _build_user_msg(query: str, book_source: str | None, memory_context: str) -> str:
    msg = query
    if book_source:
        msg = f"[当前书籍来源：{book_source}]\n\n{msg}"
    if memory_context:
        msg += f"\n\n[历史阅读记录（仅供参考）]\n{memory_context}"
    return msg


class DeepReadAgent:
    """无状态 ReAct agent。每次 run() 构建独立工具闭包，支持多用户并发。

    HybridRetriever 作为类级缓存保存，BM25 索引跨请求复用。
    上传新书后调用 invalidate_retriever() 清除 BM25 缓存。
    """

    def __init__(
        self,
        *,
        store: ChromaStore | None = None,
        collection_name: str | None = None,
        llm=None,
        config: DeepReadConfig | None = None,
    ) -> None:
        self.store = store or ChromaStore()
        self.llm = llm or get_llm(temperature=0.2)
        self.config = config or DeepReadConfig()
        self._collection_name = collection_name or self.store.collection_name
        self._retriever: HybridRetriever | None = None

    def _get_retriever(self) -> HybridRetriever:
        """懒初始化 HybridRetriever，上传新书后调用 invalidate_retriever() 重建。"""
        if self._retriever is None:
            hybrid_cfg = self.config.hybrid or HybridConfig(
                fetch_k=self.config.fetch_k,
                final_k=self.config.k,
            )
            self._retriever = HybridRetriever(
                store=self.store,
                collection_name=self._collection_name,
                config=hybrid_cfg,
                llm=self.llm,
            )
        return self._retriever

    def invalidate_retriever(self) -> None:
        """上传新书后调用，清除 BM25 缓存，下次检索时自动重建索引。"""
        if self._retriever is not None:
            self._retriever.invalidate_bm25()
            logger.info("BM25 缓存已清除")

    def _search_books_impl(
        self,
        search_query: str,
        scope: str,
        book_source: str | None,
        chapter: str | None,
    ) -> list[Document]:
        """Core retrieval logic for search_books tool. Extracted for testability."""
        filter_ = {"source": book_source} if (scope == "current_book" and book_source) else None
        effective_query = f"{chapter} {search_query}" if chapter else search_query
        return self._get_retriever().search(effective_query, filter=filter_)

    def _build(self, *, book_source: str | None, book_id: str):
        """Build a react_agent with a single search_books tool closure. Returns (agent, current_docs)."""
        current_docs: list[Document] = []
        config = self.config

        @tool
        def search_books(
            search_query: str,
            scope: str = "current_book",
            chapter: str | None = None,
        ) -> str:
            """搜索书库内容。
            search_query: 检索关键词或问题。
            scope: "current_book"（默认，当前书）或 "all_books"（跨全书库，用于跨书对比）。
            chapter: 章节名称，非空时聚焦该章节内容（用于章节摘要）。
            """
            docs = self._search_books_impl(search_query, scope, book_source, chapter)

            logger.info(
                "search_books query=%r scope=%r chapter=%r hits=%d",
                search_query, scope, chapter, len(docs),
            )
            if not docs:
                return "未找到相关内容，请尝试换一种关键词。"

            seen = {d.page_content[:100] for d in current_docs}
            for d in docs:
                key = d.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    current_docs.append(d)

            display = docs[: max(1, config.max_evidence)]
            blocks: list[str] = []
            for i, d in enumerate(display, 1):
                meta = d.metadata or {}
                title = meta.get("book_title") or "未知书名"
                location = meta.get("section_title") or meta.get("chapter_title") or ""
                blocks.append(
                    f"[证据{i}] 书名：{title}  章节：{location}\n"
                    + (d.page_content or "").strip()[:600]
                )
            return sep.join(blocks)

        from langgraph.prebuilt import create_react_agent
        react_agent = create_react_agent(
            self.llm,
            [search_books],
            prompt=_SYSTEM_PROMPT,
        )
        return react_agent, current_docs

    def run(
        self,
        *,
        query: str,
        book_source: str | None = None,
        book_id: str = "",
        memory_context: str = "",
        user_id: str = "default",
        history: list[dict] | None = None,
    ) -> DeepReadResult:
        logger.info("run user=%r query=%r source=%r history_len=%d", user_id, query, book_source, len(history or []))
        react_agent, current_docs = self._build(book_source=book_source, book_id=book_id)
        user_msg = _build_user_msg(query, book_source, memory_context)
        history_msgs = [(m["role"], m["content"]) for m in (history or [])]
        result = react_agent.invoke(
            {"messages": [*history_msgs, ("user", user_msg)]},
            config={"recursion_limit": 8},
        )
        answer = result["messages"][-1].content
        citations = build_citations(current_docs)
        return DeepReadResult(
            answer=answer,
            citations=citations,
            retrieved_docs=list(current_docs),
        )

    async def astream_events(
        self,
        *,
        query: str,
        book_source: str | None = None,
        book_id: str = "",
        memory_context: str = "",
        user_id: str = "default",
        history: list[dict] | None = None,
    ) -> AsyncGenerator[tuple[str, object], None]:
        """Async generator yielding (event_type, data) for SSE streaming.

        Yields:
            ("token", str)  — incremental text chunk
            ("done", dict)  — final metadata: citations, docs_count
        """
        logger.info("stream user=%r query=%r source=%r history_len=%d", user_id, query, book_source, len(history or []))
        react_agent, current_docs = self._build(book_source=book_source, book_id=book_id)
        user_msg = _build_user_msg(query, book_source, memory_context)
        history_msgs = [(m["role"], m["content"]) for m in (history or [])]

        async for event in react_agent.astream_events(
            {"messages": [*history_msgs, ("user", user_msg)]},
            config={"recursion_limit": 8},
            version="v2",
        ):
            etype = event["event"]
            if etype == "on_tool_start":
                yield "tool", event.get("name", "tool")
            elif etype == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                # Skip tool-call chunks; only forward plain text
                if chunk.tool_call_chunks:
                    continue
                content = chunk.content
                if isinstance(content, str) and content:
                    yield "token", content
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if text:
                                yield "token", text

        citations = build_citations(current_docs)
        yield "done", {
            "citations": [c.__dict__ for c in citations],
            "docs_count": len(current_docs),
        }


__all__ = ["DeepReadAgent", "DeepReadResult", "DeepReadConfig"]
