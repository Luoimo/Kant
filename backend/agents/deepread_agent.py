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


_SYSTEM_PROMPT_BASE = """\
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

_GRAPH_AWARE_APPENDIX = """\
Graph-aware 证据融合规则（当 search_book_content 返回图结构块时生效）：
1. 你会同时收到两类证据：
   - [证据i]：向量检索命中的原文片段（可直接引用的文本证据）。
   - [图检索子图]：图谱结构信息（种子节点/扩散节点/关联章节/关系路径），用于关系推理与上下文补全。
2. 证据职责分工：
   - 事实性表述、原话、细节优先依赖 [证据i]。
   - 关系、依赖、层级、角色互动优先依赖 [图检索子图]。
3. 若两类证据不一致：
   - 明确指出差异；
   - 不要编造不存在于任一证据源的信息；
   - 优先采用可被原文片段直接验证的结论。
4. 输出时若使用了图结构推理，需单独给出「图结构依据」小节，避免把结构推理伪装成原文引用。
"""
_SYSTEM_PROMPT = _SYSTEM_PROMPT_BASE + _GRAPH_AWARE_APPENDIX


def _build_user_msg(query: str, book_source: str | None, memory_context: str) -> str:
    msg = query
    if book_source:
        msg = f"[当前书籍来源：{book_source}]\n\n{msg}"
    if memory_context:
        msg += f"\n\n[历史阅读记录（仅供参考）]\n{memory_context}"
    return msg


class DeepReadAgent:
    """无状态 ReAct agent。每次 run() 构建独立工具闭包，支持多用户并发。"""

    def __init__(
        self,
        *,
        store: ChromaStore | None = None,
        collection_name: str | None = None,
        llm=None,
        config: DeepReadConfig | None = None,
        note_vector_store=None,
    ) -> None:
        self.store = store or ChromaStore()
        self.llm = llm or get_llm(temperature=0.2)
        self.config = config or DeepReadConfig()
        self._collection_name = collection_name or self.store.collection_name
        self._note_vector_store = note_vector_store

    def _build(self, *, book_source: str | None, book_id: str):
        """Build a react_agent with bound tool closures. Returns (agent, current_docs)."""
        current_docs: list[Document] = []
        store = self.store
        config = self.config
        collection_name = self._collection_name
        llm = self.llm

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
            lines = ["[图检索子图]"]
            if seeds:
                lines.append("种子节点: " + " / ".join(seeds))
            if expanded:
                lines.append("扩散节点: " + " / ".join(expanded))
            if chapters:
                lines.append("关联章节: " + " / ".join(chapters))
            if paths:
                lines.append("关系路径: " + " | ".join(paths))
            return "\n".join(lines)

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
            if graph_block:
                blocks.append(graph_block)
            logger.info("blocks=%r", blocks)
            return sep.join(blocks)

        @tool
        def search_past_notes(query: str) -> str:
            """检索用户的历史读书笔记。
            当用户询问“我之前记过什么”、“关于某某概念我以前有什么想法”，或你需要跨书串联知识时调用。
            """
            if not self._note_vector_store:
                return "笔记系统未启用，无法检索。"
            try:
                results = self._note_vector_store.search_similar(text=query, exclude_book="", top_k=3)
                if not results:
                    return "未在历史笔记中找到相关记录。"
                lines = []
                for i, r in enumerate(results, 1):
                    book = r.get("book_title", "未知书籍")
                    summary = r.get("question_summary", "")
                    concepts = ", ".join(r.get("concepts", []))
                    date = r.get("date", "")[:10]
                    lines.append(f"[笔记{i}] 《{book}》({date}): {summary} (涉及概念: {concepts})")
                return "找到以下历史笔记记录：\n" + "\n".join(lines)
            except Exception as e:
                return f"笔记检索失败：{e}"

        from langgraph.prebuilt import create_react_agent
        react_agent = create_react_agent(
            llm,
            [search_book_content, search_past_notes],
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
    ) -> DeepReadResult:
        logger.info("run user=%r query=%r source=%r", user_id, query, book_source)
        react_agent, current_docs = self._build(book_source=book_source, book_id=book_id)
        user_msg = _build_user_msg(query, book_source, memory_context)
        result = react_agent.invoke(
            {"messages": [("user", user_msg)]},
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
    ) -> AsyncGenerator[tuple[str, object], None]:
        """Async generator yielding (event_type, data) for SSE streaming.

        Yields:
            ("token", str)  — incremental text chunk
            ("done", dict)  — final metadata: citations, docs_count
        """
        logger.info("stream user=%r query=%r source=%r", user_id, query, book_source)
        react_agent, current_docs = self._build(book_source=book_source, book_id=book_id)
        user_msg = _build_user_msg(query, book_source, memory_context)
        logger.info("deepread agent prompt(system):\n%s", _SYSTEM_PROMPT)
        logger.info("deepread agent prompt(user):\n%s", user_msg)

        async for event in react_agent.astream_events(
            {"messages": [("user", user_msg)]},
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
