from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
import sys

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage

from backend.config import get_settings
from backend.llm.openai_client import get_llm, build_messages_context
from backend.rag.chroma.chroma_store import ChromaStore
from backend.rag.retriever import HybridConfig, HybridRetriever
from backend.storage.note_storage import NoteStorage, LocalNoteStorage
from backend.xai.citation import Citation, build_citations

sep = "\n\n"

NOTE_TEMPLATES: dict[str, str] = {
    "structured": (
        "使用分级 Markdown 标题（## ### ####）和要点列表组织内容，关键概念加粗 **概念**，"
        "标注页码引用（如：（p.XX）），末尾可选添加【待探索问题】小节。"
    ),
    "summary": (
        "输出三个固定小节：**核心论点**（1-3 句话）、**主要证据**（3-5 条要点）、**个人洞见**（1-3 句话）。"
    ),
    "qa": (
        "以问答卡片格式输出：每个知识点用「**Q：问题** / A：答案」的形式列出，适合备考复习。"
    ),
    "timeline": (
        "以时间线节点格式输出：每个节点格式为「**年份/时期** — 事件/概念描述」，按时间顺序排列。"
    ),
}

NOTE_SYSTEM_PROMPT = """你是"笔记整理助手（NoteAgent）"，擅长将书籍内容或零散文字整理成结构化笔记。

核心职责：
1. 将检索到的内容或用户提供的文字，整理成清晰的 Markdown 格式笔记。
2. 笔记应包含：分级标题、要点列表、关键概念、页码引用（如有）、待探索问题（可选）。
3. 重点在于"归纳与结构化"，而非一问一答。
4. 如果内容不足，如实说明，不要编造书中不存在的内容。
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
        note_storage: NoteStorage | None = None,
    ) -> None:
        if store is None:
            settings = get_settings()
            store = ChromaStore()

        self.store = store
        self.llm = llm or get_llm(temperature=0.3)
        self.k = k
        self.note_storage = note_storage  # None = no persistence (backward compat)

        # HybridRetriever 在 __init__ 构建一次，避免每次 run() 重建 BM25 索引
        self._retriever = HybridRetriever(
            store=self.store,
            collection_name=self.store.collection_name,
            config=HybridConfig(fetch_k=20, final_k=k),
            llm=self.llm,
        )

    def run(
        self,
        *,
        query: str,
        book_source: str | None = None,
        raw_text: str | None = None,
        memory_context: str = "",
        notes_messages: list[AnyMessage] | None = None,
        action: Literal["new", "edit", "extend"] = "new",
        notes_format: Literal["structured", "summary", "qa", "timeline"] = "structured",
        storage_path: str | None = None,
    ) -> NoteResult:
        # action=edit/extend 但没有 storage_path → 降级为 new
        if action in ("edit", "extend") and not storage_path:
            action = "new"

        # 路径 1：edit/extend — 加载已有笔记（load 失败时降级为 new）
        if action in ("edit", "extend") and storage_path and self.note_storage:
            try:
                existing_note = self.note_storage.load(storage_path)
            except Exception as e:
                print(f"[NoteAgent] load failed, degrading to new: {e}", file=sys.stdout)
                action = "new"
            else:
                return self._modify_note(
                    query=query,
                    existing_note=existing_note,
                    action=action,
                    notes_format=notes_format,
                    memory_context=memory_context,
                    notes_messages=notes_messages,
                )

        # 路径 2：raw_text-only（无 book_source，跳过检索）
        if raw_text and not book_source:
            print(f"[NoteAgent] raw_text mode, len={len(raw_text)}", file=sys.stdout)
            answer = self._synthesize_notes(query, [], raw_text, memory_context=memory_context,
                                            notes_messages=notes_messages, notes_format=notes_format)
            return NoteResult(answer=answer, citations=[], retrieved_docs=[])

        # 路径 3：new — 混合检索 + 生成（有 book_source 或无 raw_text）
        filter_ = {"source": book_source} if book_source else None
        docs = self._retriever.search(query, filter=filter_)
        # 无检索结果且也无 raw_text → 返回提示
        if not docs and not raw_text:
            return NoteResult(
                answer="本地书库没有检索到相关内容，无法生成笔记。请先将相关 EPUB 入库，或提供待整理的原文。",
                citations=[],
                retrieved_docs=[],
            )
        citations = build_citations(docs)
        print(
            f"[NoteAgent] query={query!r}, book_source={book_source!r}, "
            f"action={action}, hits={len(docs)}",
            file=sys.stdout,
        )
        answer = self._synthesize_notes(
            query, docs, raw_text,
            memory_context=memory_context,
            notes_messages=notes_messages,
            notes_format=notes_format,
        )
        return NoteResult(answer=answer, citations=citations, retrieved_docs=docs)

    # ------------------------------------------------------------------

    def _modify_note(
        self,
        *,
        query: str,
        existing_note: str,
        action: Literal["edit", "extend"],
        notes_format: Literal["structured", "summary", "qa", "timeline"],
        memory_context: str,
        notes_messages: list[AnyMessage] | None,
    ) -> NoteResult:
        verb = "修改" if action == "edit" else "扩展/补充"
        user_prompt = (
            f"用户请求（{verb}）：{query}\n\n"
            f"以下是已有笔记的完整内容：\n\n{existing_note}\n\n"
            f"请根据用户请求对上面的笔记进行{verb}，保持 Markdown 格式一致。"
            f"\n\n输出格式要求：{NOTE_TEMPLATES.get(notes_format, NOTE_TEMPLATES['structured'])}"
        )
        system = NOTE_SYSTEM_PROMPT
        if memory_context:
            system += "\n\n【用户历史阅读记录（仅供参考）】\n" + memory_context

        messages_ctx = build_messages_context(notes_messages)
        msg = self.llm.invoke(messages_ctx + [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ])
        return NoteResult(
            answer=getattr(msg, "content", str(msg)),
            citations=[],
            retrieved_docs=[],
        )

    def _synthesize_notes(
        self,
        query: str,
        docs: list[Document],
        raw_text: str | None = None,
        memory_context: str = "",
        notes_messages: list[AnyMessage] | None = None,
        notes_format: Literal["structured", "summary", "qa", "timeline"] = "structured",
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

        label = (
            "【书库检索片段 + 用户文本】：" if docs and raw_text
            else "【书库检索片段】：" if docs
            else "【用户提供文本】："
        )
        format_instruction = NOTE_TEMPLATES.get(notes_format, NOTE_TEMPLATES["structured"])

        user_prompt = (
            f"用户笔记整理请求：\n{query}\n\n"
            f"请将以下内容整理成结构化 Markdown 笔记：\n\n"
            f"{label}\n{sep.join(content_blocks)}\n\n"
            f"整理要求：\n{format_instruction}\n"
            "只整理实际出现的内容，不要添加书中没有的信息。"
        )
        system = NOTE_SYSTEM_PROMPT
        if memory_context:
            system += "\n\n【用户历史阅读记录（仅供参考）】\n" + memory_context

        messages_ctx = build_messages_context(notes_messages)
        msg = self.llm.invoke(messages_ctx + [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ])
        return getattr(msg, "content", str(msg))



def notes_node(
    state: dict[str, Any],
    *,
    agent: NoteAgent,
    deps: Any = None,
    thread_id: str = "default",
) -> dict[str, Any]:
    """
    模块级节点函数，被 orchestrator 闭包调用。
    返回 delta dict（包含 compound_context, notes_messages, notes_last_output）。
    """
    query: str = state.get("notes_query", "") or state.get("user_input", "")
    book_source: str | None = state.get("notes_book_source") or state.get("book_source")
    memory_context: str = state.get("memory_context", "") or ""
    notes_messages: list[AnyMessage] = state.get("notes_messages") or []
    action = state.get("action") or "new"
    last_output: dict = state.get("notes_last_output") or {}
    storage_path: str | None = last_output.get("storage_path")

    result = agent.run(
        query=query,
        book_source=book_source,
        memory_context=memory_context,
        notes_messages=notes_messages,
        action=action,
        storage_path=storage_path,
    )
    content = result.answer

    # 持久化笔记
    storage_path_out: str | None = None
    note_id = f"note_{thread_id}_{int(datetime.now(tz=timezone.utc).timestamp())}"
    note_storage = getattr(deps, "note_storage", None) if deps else None
    if not note_storage:
        note_storage = agent.note_storage
    if note_storage:
        storage_path_out = note_storage.save(content, note_id)

    # 提取书名
    book_title = ""
    if result.retrieved_docs:
        book_title = (result.retrieved_docs[0].metadata or {}).get("book_title", "")

    existing_ctx = state.get("compound_context") or ""
    new_ctx = (existing_ctx + f"\n\n[笔记结果]\n{content[:1500]}").strip()

    return {
        "answer": content,
        "citations": result.citations,
        "retrieved_docs_count": len(result.retrieved_docs),
        "notes_messages": [AIMessage(content=content)],
        "notes_last_output": {
            "note_id": note_id,
            "book_title": book_title,
            "topics": [],
            "storage_path": storage_path_out or "",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        },
        "compound_context": new_ctx,
    }


__all__ = ["NoteAgent", "NoteResult", "notes_node"]
