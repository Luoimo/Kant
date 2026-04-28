from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)
from typing import Any, AsyncGenerator

from langchain_core.documents import Document
from langchain_core.tools import tool

from graph.neo4j_store import get_neo4j_store
from llm.openai_client import get_llm
from prompts import get_prompts
from rag.chroma.chroma_store import ChromaStore
from rag.retriever import HybridConfig, HybridRetriever
from xai.citation import Citation, build_citations

sep = "\n\n"
_CHINESE_TEXT_RE = re.compile(r"[\u4e00-\u9fff]+")
_ENGLISH_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
_STOP_TERMS = {
    "角色", "分析", "介绍", "评价", "解读", "理解", "看法",
    "人物", "为什么", "怎么", "如何", "关系", "结局",
    "意义", "作用", "特点", "性格", "是什么", "哪些",
}


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
    enable_graph_retrieval: bool = True
    graph_seed_top_k: int = 6
    graph_expand_top_k: int = 10
    graph_chunk_k: int = 24


_SYSTEM_PROMPT_BASE = None  # kept for backward compat; real prompts come from prompts module
_GRAPH_AWARE_APPENDIX = None
_SYSTEM_PROMPT = None


def _build_system_msg(
    book_title: str | None,
    book_source: str | None,
    memory_context: str,
    selected_text: str | None,
    current_chapter: str | None,
    *,
    locale: str | None = None,
) -> str:
    """Build dynamic context to inject into the system prompt instead of user message."""
    p = get_prompts(locale).deepread
    parts = []
    if book_title:
        parts.append(p.ctx_current_book.format(title=book_title))
    if book_source:
        parts.append(p.ctx_book_source.format(source=book_source))
    if current_chapter:
        parts.append(p.ctx_current_chapter.format(chapter=current_chapter))
    if selected_text:
        parts.append(p.ctx_selected_text.format(text=selected_text))
    if memory_context:
        parts.append(p.ctx_memory.format(memory=memory_context))

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

    def _build(self, *, book_source: str | None, book_id: str, memory=None, sys_msg: str = "", locale: str | None = None):
        """Build a react_agent with bound tool closures. Returns (agent, current_docs)."""
        current_docs: list[Document] = []
        store = self.store
        config = self.config
        collection_name = self._collection_name
        llm = self.llm
        prompts = get_prompts(locale).deepread

        # Compose the full system prompt from the locale-specific bundle.
        full_system_prompt = prompts.system_base + prompts.graph_aware_appendix
        if sys_msg:
            full_system_prompt += f"\n\n[System Context Update]\n{sys_msg}"

        def _clean_chinese_phrase(text: str) -> list[str]:
            candidates: set[str] = set()
            text = text.strip()
            if len(text) < 2:
                return []

            candidates.add(text)

            cleaned = text
            for stop in sorted(_STOP_TERMS, key=len, reverse=True):
                cleaned = cleaned.replace(stop, "")
            cleaned = cleaned.strip()
            if len(cleaned) >= 2:
                candidates.add(cleaned)

            return list(candidates)

        def _extract_graph_terms(query: str) -> list[str]:
            query = (query or "").strip()
            if not query:
                return []

            mentions: list[str] = []
            for m in _ENGLISH_TERM_RE.findall(query):
                mentions.append(m.strip())
            for chunk in _CHINESE_TEXT_RE.findall(query):
                mentions.extend(_clean_chinese_phrase(chunk))

            def _score(x: str) -> tuple[int, str]:
                return (len(x), x)

            uniq: list[str] = []
            seen: set[str] = set()
            for m in sorted(mentions, key=_score):
                key = m.lower()
                if key not in seen and len(m) >= 2:
                    seen.add(key)
                    uniq.append(m)

            return uniq[:20]

        def _graph_retrieve_subgraph(search_query: str) -> dict[str, Any]:
            if not config.enable_graph_retrieval or not book_id:
                return {"seed_entities": [], "expanded_entities": [], "chapter_titles": [], "reasoning_paths": []}
            terms = _extract_graph_terms(search_query)
            if not terms:
                return {"seed_entities": [], "expanded_entities": [], "chapter_titles": [], "reasoning_paths": []}
            return get_neo4j_store().graph_retrieve_chunks(
                book_id=book_id,
                query_terms=terms,
                seed_top_k=config.graph_seed_top_k,
                expand_top_k=config.graph_expand_top_k,
                chapter_limit=config.graph_chunk_k,
            )

        def _build_graph_block(payload: dict[str, Any]) -> str:
            seeds = [str(x) for x in (payload.get("seed_entities") or []) if str(x).strip()][:8]
            expanded = [str(x) for x in (payload.get("expanded_entities") or []) if str(x).strip()][:10]
            chapters = [str(x) for x in (payload.get("chapter_titles") or []) if str(x).strip()][:8]
            paths = [str(x) for x in (payload.get("reasoning_paths") or []) if str(x).strip()][:12]
            if not (seeds or expanded or chapters or paths):
                return ""
            lines = [prompts.graph_block_title]
            if seeds:
                lines.append(prompts.graph_seeds + " / ".join(seeds))
            if expanded:
                lines.append(prompts.graph_expanded + " / ".join(expanded))
            if chapters:
                lines.append(prompts.graph_chapters + " / ".join(chapters))
            if paths:
                lines.append(prompts.graph_paths + " | ".join(paths))
            return "\n".join(lines)

        @tool
        def search_book_content(search_query: str) -> str:
            """Search the user's local library for textual evidence."""
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
            vector_docs = retriever.search(search_query, filter=filter_)
            graph_payload = _graph_retrieve_subgraph(search_query)
            limit = max(config.max_evidence * 2, config.k)
            docs = vector_docs[:limit]

            logger.info(
                "search query=%r source=%r vector_hits=%d graph_nodes=%d merged=%d seeds=%s expanded_entities=%s expanded_pairs=%s",
                search_query,
                book_source,
                len(vector_docs),
                len((graph_payload.get("expanded_entities") or [])),
                len(docs),
                (graph_payload.get("seed_entities") or [])[:6],
                (graph_payload.get("expanded_entities") or [])[:6],
                (graph_payload.get("expanded_pairs") or [])[:6],
            )
            graph_block = _build_graph_block(graph_payload)
            if not docs and not graph_block:
                return prompts.no_results

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
                title = meta.get("book_title") or prompts.evidence_unknown_book
                location = meta.get("section_title") or meta.get("chapter_title") or ""
                header = prompts.evidence_header.format(i=i, title=title, location=location)
                blocks.append(header + "\n" + (d.page_content or "").strip()[:600])
            if graph_block:
                blocks.append(graph_block)
            logger.info("blocks=%r", blocks)
            return sep.join(blocks)

        search_book_content.description = prompts.tool_search_book_desc

        @tool
        def search_past_notes(query: str) -> str:
            """Retrieve user's past notes."""
            from agents.obsidian_tools import search_vault_for_concept
            return search_vault_for_concept.invoke(query)

        search_past_notes.description = prompts.tool_search_notes_desc

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

    def clear_chat_history(
        self,
        *,
        book_id: str = "",
        user_id: str = "default",
        thread_id: str = "default",
    ) -> None:
        """Clear chat history for a given user and book."""
        from pathlib import Path
        import sqlite3
        
        db_path = Path("data/chat_history.db")
        if not db_path.exists():
            return
            
        actual_thread = f"{user_id}_{book_id}" if book_id else thread_id
        
        try:
            with sqlite3.connect(db_path, check_same_thread=False) as conn:
                conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (actual_thread,))
                conn.execute("DELETE FROM writes WHERE thread_id = ?", (actual_thread,))
                conn.commit()
                logger.info("Successfully cleared chat history for thread %s", actual_thread)
        except Exception as e:
            logger.error("Failed to clear chat history: %s", e)

    def add_ai_message(
        self,
        content: str,
        *,
        book_id: str = "",
        user_id: str = "default",
        thread_id: str = "default",
        locale: str | None = None,
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

                react_agent, _ = self._build(book_source=None, book_id=book_id, memory=memory, locale=locale)
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
        locale: str | None = None,
    ) -> DeepReadResult:
        from langgraph.checkpoint.sqlite import SqliteSaver
        from pathlib import Path
        import sqlite3

        logger.info("run user=%r query=%r source=%r locale=%r", user_id, query, book_source, locale)

        db_path = Path("data/chat_history.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Use a deterministic thread_id based on user and book so chat history is maintained
        actual_thread = f"{user_id}_{book_id}" if book_id else thread_id

        # 提取动态上下文并组装为系统指令
        sys_msg = _build_system_msg(
            book_title=book_id,
            book_source=book_source,
            memory_context=memory_context,
            selected_text=selected_text,
            current_chapter=current_chapter,
            locale=locale,
        )

        with sqlite3.connect(db_path, check_same_thread=False) as conn:
            memory = SqliteSaver(conn)
            memory.setup()

            react_agent, current_docs = self._build(
                book_source=book_source, book_id=book_id, memory=memory,
                sys_msg=sys_msg, locale=locale,
            )

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
        locale: str | None = None,
    ) -> AsyncGenerator[tuple[str, object], None]:
        """Async generator yielding (event_type, data) for SSE streaming."""
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from pathlib import Path
        import aiosqlite

        logger.info("stream user=%r query=%r source=%r locale=%r", user_id, query, book_source, locale)

        db_path = Path("data/chat_history.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)

        actual_thread = f"{user_id}_{book_id}" if book_id else thread_id

        sys_msg = _build_system_msg(
            book_title=book_id,
            book_source=book_source,
            memory_context=memory_context,
            selected_text=selected_text,
            current_chapter=current_chapter,
            locale=locale,
        )

        async with aiosqlite.connect(db_path) as conn:
            memory = AsyncSqliteSaver(conn)
            memory.setup()

            react_agent, current_docs = self._build(
                book_source=book_source, book_id=book_id, memory=memory,
                sys_msg=sys_msg, locale=locale,
            )

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
