from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import AsyncGenerator

from langchain_core.documents import Document
from langchain_core.tools import tool

from llm.openai_client import get_llm
from rag.chroma.chroma_store import ChromaStore
from rag.retriever import HybridConfig, HybridRetriever
from xai.citation import Citation, build_citations

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
- search_past_notes    : 检索用户的历史读书笔记。当用户询问以前的思考、或者需要跨书串联知识时调用。

工作原则（防幻觉与客观性要求）：
1. 书中内容问答 — 必须有 search_book_content 的证据支撑。如果检索到的证据不足以回答问题，你必须明确回答『根据书本内容，无法直接回答该问题』或『书中未提及相关内容』，绝对不要使用外部知识编造或猜测书中事实（Hallucination Mitigation）。
2. 解释概念与跨书对比（公平性与偏见控制） — 在解释哲学概念或做跨书观点对比时，必须体现多元化（Diversity），涵盖不同文化背景、流派或视角的观点，避免单一维度的偏见（Bias Mitigation），确保内容客观公平（Fairness）。

输出格式：
- 内容问答：结构化回答 + 末尾「引用」小节（书名·章节）
"""


def _build_system_msg(
    book_title: str | None,
    book_source: str | None,
    memory_context: str,
    selected_text: str | None,
    current_chapter: str | None,
) -> str:
    """Build dynamic context to inject into the system prompt instead of user message."""
    parts = []
    if book_title:
        parts.append(f"【当前阅读书籍】：《{book_title}》")
    if book_source:
        parts.append(f"【当前书籍来源】：{book_source}")
    if current_chapter:
        parts.append(f"【当前阅读章节】：{current_chapter}")
    if selected_text:
        parts.append(f"【用户当前划选的原文片段】（用户的问题可能针对这段话）：\n{selected_text}")
    if memory_context:
        parts.append(f"【用户历史阅读记录】（仅供参考，用于个性化回答）：\n{memory_context}")
        
    return "\n\n".join(parts) if parts else ""


class DeepReadAgent:
    """无状态 ReAct agent。每次 run() 构建独立工具闭包，支持多用户并发。"""

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

    def _build(self, *, book_source: str | None, book_id: str, memory=None, sys_msg: str = ""):
        """Build a react_agent with bound tool closures. Returns (agent, current_docs)."""
        current_docs: list[Document] = []
        store = self.store
        config = self.config
        collection_name = self._collection_name
        llm = self.llm
        
        # Combine the static _SYSTEM_PROMPT with any dynamic context
        full_system_prompt = _SYSTEM_PROMPT
        if sys_msg:
            full_system_prompt += f"\n\n[System Context Update]\n{sys_msg}"

        @tool
        def search_book_content(search_query: str) -> str:
            """在用户本地书库中检索原文证据片段。
            输入精简的搜索关键词；证据不足时可换关键词再调用一次。
            """
            hybrid_cfg = config.hybrid or HybridConfig(
                fetch_k=config.fetch_k,
                final_k=config.k,
            )
            retriever = HybridRetriever(
                store=store,
                collection_name=collection_name,
                config=hybrid_cfg,
                llm=llm,
            )
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
        def search_past_notes(query: str) -> str:
            """检索用户的历史读书笔记。
            当用户询问“我之前记过什么”、“关于某某概念我以前有什么想法”，或你需要跨书串联知识时调用。
            """
            from agents.obsidian_tools import search_vault_for_concept
            return search_vault_for_concept.invoke(query)

        from langgraph.prebuilt import create_react_agent
        
        react_agent = create_react_agent(
            llm,
            [search_book_content, search_past_notes],
            prompt=full_system_prompt,
            checkpointer=memory,
        )
        return react_agent, current_docs

    def get_chat_history(
        self,
        *,
        book_id: str = "",
        user_id: str = "default",
        thread_id: str = "default",
    ) -> list[dict]:
        """Fetch chat history for a given user and book."""
        from langgraph.checkpoint.sqlite import SqliteSaver
        from pathlib import Path
        import sqlite3
        from langchain_core.messages import BaseMessage
        
        db_path = Path("data/chat_history.db")
        if not db_path.exists():
            return []
            
        actual_thread = f"{user_id}_{book_id}" if book_id else thread_id
        
        try:
            with sqlite3.connect(db_path, check_same_thread=False) as conn:
                memory = SqliteSaver(conn)
                memory.setup()
                tuple_ = memory.get_tuple({"configurable": {"thread_id": actual_thread}})
                if not tuple_:
                    return []
                    
                # extract messages from the state
                state = tuple_.checkpoint.get("channel_values", {})
                messages = state.get("messages", [])
                
                history = []
                for m in messages:
                    if not isinstance(m, BaseMessage):
                        continue
                    
                    # Filter out system messages or tool calls if needed
                    if m.type in ["human", "user"]:
                        content = m.content
                        # 尝试剔除后端拼装的系统前缀，只提取真实的 user query
                        import re
                        if "【用户问题】：\n" in content:
                            content = content.split("【用户问题】：\n")[-1]
                        elif "[历史阅读记录（仅供参考）]" in content:
                            # 如果包含阅读记录，通常用户问题在最前面
                            content = content.split("\n\n[历史阅读记录（仅供参考）]")[0]
                            # 去除可能存在的书籍前缀
                            content = re.sub(r'^\[当前书籍来源：.*?\]\n\n', '', content)
                        else:
                            content = re.sub(r'^\[当前书籍来源：.*?\]\n\n', '', content)
                            content = re.sub(r'^【当前阅读书籍】：.*?\n\n', '', content)
                            content = re.sub(r'\n\n【当前阅读章节】：.*$', '', content)
                            
                        history.append({"role": "user", "content": content.strip()})
                    elif m.type in ["ai", "assistant"]:
                        # Skip if it's just a tool call with no real content
                        if not m.content and getattr(m, "tool_calls", None):
                            continue
                        history.append({"role": "ai", "content": m.content})
                        
                return history
        except Exception as e:
            logger.error("Failed to load chat history: %s", e)
            return []

    def add_ai_message(
        self,
        content: str,
        *,
        book_id: str = "",
        user_id: str = "default",
        thread_id: str = "default",
    ) -> None:
        """Manually append an AI message (e.g. from CriticAgent) to the chat history."""
        from langgraph.checkpoint.sqlite import SqliteSaver
        from pathlib import Path
        import sqlite3
        from langchain_core.messages import AIMessage
        
        db_path = Path("data/chat_history.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        actual_thread = f"{user_id}_{book_id}" if book_id else thread_id
        
        try:
            with sqlite3.connect(db_path, check_same_thread=False) as conn:
                memory = SqliteSaver(conn)
                memory.setup()
                
                react_agent, _ = self._build(book_source=None, book_id=book_id, memory=memory)
                react_agent.update_state(
                    {"configurable": {"thread_id": actual_thread}},
                    {"messages": [AIMessage(content=content)]}
                )
                logger.info("Successfully appended AI message to thread %s", actual_thread)
        except Exception as e:
            logger.error("Failed to append AI message to history: %s", e)

    def run(
        self,
        *,
        query: str,
        book_source: str | None = None,
        book_id: str = "",
        memory_context: str = "",
        user_id: str = "default",
        thread_id: str = "default",
        selected_text: str | None = None,
        current_chapter: str | None = None,
    ) -> DeepReadResult:
        from langgraph.checkpoint.sqlite import SqliteSaver
        from pathlib import Path
        import sqlite3
        
        logger.info("run user=%r query=%r source=%r", user_id, query, book_source)
        
        db_path = Path("data/chat_history.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use a deterministic thread_id based on user and book so chat history is maintained
        actual_thread = f"{user_id}_{book_id}" if book_id else thread_id
        
        # 提取动态上下文并组装为系统指令
        sys_msg = _build_system_msg(
            book_title=book_id,  # For simplistic implementation here
            book_source=book_source,
            memory_context=memory_context,
            selected_text=selected_text,
            current_chapter=current_chapter,
        )
        
        with sqlite3.connect(db_path, check_same_thread=False) as conn:
            memory = SqliteSaver(conn)
            memory.setup() # Ensure tables exist
            
            react_agent, current_docs = self._build(book_source=book_source, book_id=book_id, memory=memory, sys_msg=sys_msg)
            
            result = react_agent.invoke(
                {"messages": [("user", query)]},
                config={"configurable": {"thread_id": actual_thread}, "recursion_limit": 8},
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
        thread_id: str = "default",
        selected_text: str | None = None,
        current_chapter: str | None = None,
    ) -> AsyncGenerator[tuple[str, object], None]:
        """Async generator yielding (event_type, data) for SSE streaming."""
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from pathlib import Path
        import aiosqlite
        
        logger.info("stream user=%r query=%r source=%r", user_id, query, book_source)
        
        db_path = Path("data/chat_history.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Use a deterministic thread_id based on user and book
        actual_thread = f"{user_id}_{book_id}" if book_id else thread_id
        
        # 提取动态上下文并组装为系统指令
        sys_msg = _build_system_msg(
            book_title=book_id,
            book_source=book_source,
            memory_context=memory_context,
            selected_text=selected_text,
            current_chapter=current_chapter,
        )
        
        async with aiosqlite.connect(db_path) as conn:
            memory = AsyncSqliteSaver(conn)
            memory.setup() # Ensure tables exist
            
            react_agent, current_docs = self._build(book_source=book_source, book_id=book_id, memory=memory, sys_msg=sys_msg)

            async for event in react_agent.astream_events(
                {"messages": [("user", query)]},
                config={"configurable": {"thread_id": actual_thread}, "recursion_limit": 8},
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
