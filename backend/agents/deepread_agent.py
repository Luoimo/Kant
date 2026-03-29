from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)
from pathlib import Path
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
你是"阅读助手"，帮助用户深度理解哲学和社科书籍。

工具说明：
- search_book_content  : 在用户本地书库检索原文证据。回答书中内容问题时必须调用。
- recommend_books      : 获取用户已有书单，然后你基于自身知识推荐新书。
- get_reading_plan     : 查看当前书籍的阅读计划。
- update_reading_plan  : 保存修改后的阅读计划。修改计划时先调 get_reading_plan，\
生成新内容后再调此工具保存。

工作原则：
1. 书中内容问答 — 必须有 search_book_content 的证据支撑，不编造书中事实
2. 书籍推荐 — 先调 recommend_books 获取已有书单，再用你的知识推荐新书
3. 计划查看/修改 — get_reading_plan 读，update_reading_plan 写

输出格式：
- 内容问答：结构化回答 + 末尾「引用」小节（书名·章节）
- 书籍推荐：每本用 ### 书名（作者）开头，含推荐理由和难度
- 计划：展示完整 Markdown
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
        """Build a react_agent with bound tool closures. Returns (agent, current_docs)."""
        current_docs: list[Document] = []
        config = self.config
        retriever = self._get_retriever()  # 复用类级缓存，BM25 跨请求复用

        @tool
        def search_book_content(search_query: str) -> str:
            """在用户本地书库中检索原文证据片段。
            输入精简的搜索关键词；证据不足时可换关键词再调用一次。
            """
            filter_ = {"source": book_source} if book_source else None
            docs = retriever.search(search_query, filter=filter_)

            logger.info("search query=%r source=%r hits=%d", search_query, book_source, len(docs))
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

        @tool
        def recommend_books(topic: str = "") -> str:
            """获取用户本地书库已有书单，以便推荐时避免重复。
            调用后请基于你的训练知识为用户推荐新书。
            topic: 推荐主题或关键词（可为空）。
            """
            try:
                from backend.storage.book_catalog import get_book_catalog
                books = get_book_catalog().get_all()
                if not books:
                    return "用户本地书库为空，可自由推荐。"
                lines = [f"- 《{b['title']}》（{b['author']}）" for b in books[:15]]
                header = f"用户已有书库（共{len(books)}本），推荐时请避免重复：\n"
                suffix = "\n\n请基于你的知识，为用户推荐与话题相关的、书库中尚未有的书籍。"
                return header + "\n".join(lines) + suffix
            except Exception as e:
                return f"书库查询失败：{e}，请直接基于你的知识推荐。"

        @tool
        def get_reading_plan() -> str:
            """查看当前书籍的阅读计划。修改计划前必须先调用此工具。"""
            if not book_id:
                return "当前未打开具体书籍，无法查看计划。"
            try:
                from backend.storage.book_catalog import get_plan_catalog
                record = get_plan_catalog().get_by_book_id(book_id)
                if not record:
                    return "该书暂无阅读计划，请先在 Reader 模式中打开该书自动生成计划。"
                path = Path(record["file_path"])
                return path.read_text(encoding="utf-8") if path.exists() else "计划文件不存在。"
            except Exception as e:
                return f"加载计划失败：{e}"

        @tool
        def update_reading_plan(updated_content: str) -> str:
            """将修改后的完整阅读计划 Markdown 保存到文件。
            必须先调用 get_reading_plan 查看原计划，再生成完整新内容后调用此工具。
            updated_content: 完整的新计划 Markdown 文本。
            """
            if not book_id:
                return "当前未打开具体书籍，无法保存计划。"
            try:
                from backend.storage.book_catalog import get_plan_catalog
                record = get_plan_catalog().get_by_book_id(book_id)
                if not record:
                    return "该书暂无计划记录，无法保存。请先在 Reader 模式中初始化。"
                path = Path(record["file_path"])
                path.write_text(updated_content, encoding="utf-8")
                get_plan_catalog().touch(book_id)
                logger.info("plan updated for book_id=%r", book_id)
                return "计划已保存。"
            except Exception as e:
                return f"保存计划失败：{e}"

        from langgraph.prebuilt import create_react_agent
        react_agent = create_react_agent(
            self.llm,
            [search_book_content, recommend_books, get_reading_plan, update_reading_plan],
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
