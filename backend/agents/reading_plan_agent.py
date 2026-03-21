from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
import sys

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage

from backend.config import get_settings
from backend.llm.openai_client import get_llm
from backend.rag.chroma.chroma_store import ChromaStore
from backend.rag.retriever import HybridConfig, HybridRetriever
from backend.storage.plan_storage import PlanStorage, LocalPlanStorage
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

PLAN_TYPE_HINTS: dict[str, str] = {
    "single_deep": "针对一本书进行深度精读，输出逐章节安排和阅读目标。",
    "multi_theme": "围绕一个主题跨多本书制定交叉阅读路线，注重主题联系。",
    "research": "以研究为目的，列出各书关键章节和标注重点，适合学术场景。",
}


@dataclass(frozen=True)
class ChapterInfo:
    title: str
    estimated_chars: int

    @property
    def reading_minutes(self) -> float:
        """按中文 300 字/分钟估算阅读时间。"""
        return self.estimated_chars / 300.0


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
        plan_storage: PlanStorage | None = None,
    ) -> None:
        if store is None:
            settings = get_settings()
            store = ChromaStore()

        self.store = store
        self.llm = llm or get_llm(temperature=0.4)
        self.k = k
        self.plan_storage = plan_storage

        # HybridRetriever 一次性构建
        self._retriever = HybridRetriever(
            store=self.store,
            collection_name=self.store.collection_name,
            config=HybridConfig(fetch_k=20, final_k=k),
            llm=self.llm,
        )

    # ------------------------------------------------------------------
    # 章节结构提取
    # ------------------------------------------------------------------

    def _extract_chapter_structure(self, book_source: str) -> list[ChapterInfo]:
        """从 ChromaDB 拉取该书所有 chunks，按 chapter_title 聚合字数估算阅读时间。"""
        try:
            all_docs = self.store.get_all_documents(
                collection_name=self.store.collection_name,
                filter={"source": book_source},
            )
        except Exception as e:
            print(f"[ReadingPlanAgent] get_all_documents failed: {e}", file=sys.stdout)
            return []

        chapter_chars: dict[str, int] = {}
        for doc in all_docs:
            meta = doc.metadata or {}
            chapter = meta.get("chapter_title") or meta.get("section_title") or "未命名章节"
            chapter_chars[chapter] = chapter_chars.get(chapter, 0) + len(doc.page_content or "")

        return [
            ChapterInfo(title=title, estimated_chars=chars)
            for title, chars in chapter_chars.items()
        ]

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        query: str,
        book_source: str | None = None,
        memory_context: str = "",
        plan_messages: list[AnyMessage] | None = None,
        action: Literal["new", "edit", "extend"] = "new",
        plan_type: Literal["single_deep", "multi_theme", "research"] = "single_deep",
        storage_path: str | None = None,
        plan_progress: list[str] | None = None,
    ) -> ReadingPlanResult:
        # action=edit/extend 但没有 storage_path → 降级为 new
        if action in ("edit", "extend") and not storage_path:
            action = "new"

        # 路径 1：edit/extend — 加载已有计划并修改
        if action in ("edit", "extend") and storage_path and self.plan_storage:
            existing_plan = self.plan_storage.load(storage_path)
            return self._modify_plan(
                query=query,
                existing_plan=existing_plan,
                action=action,
                plan_type=plan_type,
                memory_context=memory_context,
                plan_messages=plan_messages,
                plan_progress=plan_progress,
            )

        # 路径 2：new — 检索 + 生成
        return self._generate_new_plan(
            query=query,
            book_source=book_source,
            memory_context=memory_context,
            plan_messages=plan_messages,
            plan_type=plan_type,
            plan_progress=plan_progress,
        )

    # ------------------------------------------------------------------

    def _modify_plan(
        self,
        *,
        query: str,
        existing_plan: str,
        action: Literal["edit", "extend"],
        plan_type: Literal["single_deep", "multi_theme", "research"],
        memory_context: str,
        plan_messages: list[AnyMessage] | None,
        plan_progress: list[str] | None,
    ) -> ReadingPlanResult:
        verb = "修改" if action == "edit" else "扩展/补充"
        progress_note = ""
        if plan_progress:
            progress_note = (
                f"\n\n【已完成章节（请在新计划中标记或跳过）】：\n"
                + "\n".join(f"- {s}" for s in plan_progress)
            )
        user_prompt = (
            f"用户请求（{verb}计划）：{query}\n\n"
            f"以下是当前阅读计划的完整内容：\n\n{existing_plan}{progress_note}\n\n"
            f"请根据用户请求对上面的计划进行{verb}，保持 Markdown 格式一致。"
        )
        system = PLAN_SYSTEM_PROMPT
        if memory_context:
            system += "\n\n【用户历史阅读记录（仅供参考）】\n" + memory_context

        history = self._build_messages_context(plan_messages)
        msg = self.llm.invoke(
            history + [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]
        )
        return ReadingPlanResult(
            answer=getattr(msg, "content", str(msg)),
            citations=[],
            retrieved_docs=[],
        )

    def _generate_new_plan(
        self,
        *,
        query: str,
        book_source: str | None,
        memory_context: str,
        plan_messages: list[AnyMessage] | None,
        plan_type: Literal["single_deep", "multi_theme", "research"],
        plan_progress: list[str] | None,
    ) -> ReadingPlanResult:
        # 获取书库书目
        available_sources: list[str] = []
        try:
            available_sources = self.store.list_sources()
        except Exception as e:
            print(f"[ReadingPlanAgent] list_sources failed: {e}", file=sys.stdout)

        # 检索相关 chunks
        filter_ = {"source": book_source} if book_source else None
        docs = self._retriever.search(query, filter=filter_) if (book_source or available_sources) else []
        citations = build_citations(docs)

        # 如果指定了书源，提取章节结构
        chapters: list[ChapterInfo] = []
        if book_source:
            chapters = self._extract_chapter_structure(book_source)

        print(
            f"[ReadingPlanAgent] query={query!r}, book_source={book_source!r}, "
            f"plan_type={plan_type}, available={len(available_sources)}, hits={len(docs)}, "
            f"chapters={len(chapters)}",
            file=sys.stdout,
        )

        answer = self._generate_plan(
            query=query,
            docs=docs,
            available_sources=available_sources,
            book_source=book_source,
            chapters=chapters,
            memory_context=memory_context,
            plan_messages=plan_messages,
            plan_type=plan_type,
            plan_progress=plan_progress,
        )
        return ReadingPlanResult(answer=answer, citations=citations, retrieved_docs=docs)

    def _generate_plan(
        self,
        query: str,
        docs: list[Document],
        available_sources: list[str],
        book_source: str | None,
        chapters: list[ChapterInfo],
        memory_context: str = "",
        plan_messages: list[AnyMessage] | None = None,
        plan_type: Literal["single_deep", "multi_theme", "research"] = "single_deep",
        plan_progress: list[str] | None = None,
    ) -> str:
        context_parts: list[str] = []

        if available_sources:
            context_parts.append(
                "【书库中的可用书目】：\n"
                + "\n".join(f"- {s}" for s in available_sources[:20])
            )

        if chapters:
            chapter_lines = []
            for ch in chapters:
                mins = ch.reading_minutes
                time_str = f"约 {mins:.0f} 分钟" if mins < 60 else f"约 {mins/60:.1f} 小时"
                chapter_lines.append(f"- {ch.title}（{time_str}）")
            context_parts.append(
                f"【《{book_source}》章节结构与预估阅读时间】：\n"
                + "\n".join(chapter_lines)
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

        type_hint = PLAN_TYPE_HINTS.get(plan_type, "")
        progress_note = ""
        if plan_progress:
            progress_note = (
                "\n\n【已完成章节（请在计划中标记为已读或跳过）】：\n"
                + "\n".join(f"- {s}" for s in plan_progress)
            )

        user_prompt = (
            f"用户阅读计划请求：\n{query}\n\n"
            f"计划类型：{type_hint}\n\n"
            f"{f'指定书目：{book_source}' if book_source else ''}\n\n"
            f"可供参考的书库信息：\n{context_str}{progress_note}\n\n"
            "请根据以上信息，制定一份切实可行的 Markdown 格式阅读计划。\n"
            "计划要包含：阅读目标、书单/章节安排、每日或每周时间表、阅读建议。\n"
            "时间估算应合理（中文约 300 字/分钟，英文约 200 词/分钟），不要过于乐观。"
        )

        system = PLAN_SYSTEM_PROMPT
        if memory_context:
            system += "\n\n【用户历史阅读记录（仅供参考）】\n" + memory_context

        history = self._build_messages_context(plan_messages)
        msg = self.llm.invoke(
            history + [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]
        )
        return getattr(msg, "content", str(msg))

    @staticmethod
    def _build_messages_context(plan_messages: list[AnyMessage] | None) -> list[dict]:
        if not plan_messages:
            return []
        history = []
        for m in plan_messages[-4:]:
            role = "assistant" if getattr(m, "type", "") == "ai" else "user"
            history.append({"role": role, "content": getattr(m, "content", "")})
        return history


def plan_node(
    state: dict[str, Any],
    *,
    agent: ReadingPlanAgent,
    deps: Any = None,
    thread_id: str = "default",
) -> dict[str, Any]:
    """
    模块级节点函数，被 orchestrator 闭包调用。
    返回 delta dict（包含 compound_context, plan_messages, plan_last_output, plan_progress）。
    """
    query: str = state.get("plan_query", "") or state.get("user_input", "")
    book_source: str | None = state.get("plan_book_source") or state.get("book_source")
    memory_context: str = state.get("memory_context", "") or ""
    plan_messages: list[AnyMessage] = state.get("plan_messages") or []
    action = state.get("action") or "new"
    plan_type = state.get("plan_type") or "single_deep"
    plan_progress: list[str] = state.get("plan_progress") or []
    last_output: dict = state.get("plan_last_output") or {}
    storage_path: str | None = last_output.get("storage_path")

    result = agent.run(
        query=query,
        book_source=book_source,
        memory_context=memory_context,
        plan_messages=plan_messages,
        action=action,
        plan_type=plan_type,
        storage_path=storage_path,
        plan_progress=plan_progress,
    )
    content = result.answer

    # 持久化计划
    storage_path_out: str | None = None
    plan_id = f"plan_{thread_id}_{int(datetime.now(tz=timezone.utc).timestamp())}"
    plan_storage = getattr(deps, "plan_storage", None) if deps else None
    if not plan_storage:
        plan_storage = agent.plan_storage
    if plan_storage:
        storage_path_out = plan_storage.save(content, plan_id)

    # 检测新完成章节（当 is_progress_update 模式下，提取 query 中的章节名）
    newly_completed: list[str] = []
    # 若用户输入包含"读完了/已读"，从 query 中提取章节名
    import re
    match = re.search(r"[《【]?(.+?)[》】]?\s*(?:我)?(?:读完了|已读|看完了|完成了)", query)
    if match:
        newly_completed = [match.group(1).strip()]

    existing_ctx = state.get("compound_context") or ""
    new_ctx = (existing_ctx + f"\n\n[计划结果]\n{content[:500]}").strip()

    # 提取涉及书名
    book_titles: list[str] = []
    if result.retrieved_docs:
        seen: set[str] = set()
        for d in result.retrieved_docs:
            t = (d.metadata or {}).get("book_title", "")
            if t and t not in seen:
                seen.add(t)
                book_titles.append(t)

    return {
        "answer": content,
        "citations": result.citations,
        "retrieved_docs_count": len(result.retrieved_docs),
        "plan_messages": [AIMessage(content=content)],
        "plan_last_output": {
            "plan_id": plan_id,
            "book_titles": book_titles,
            "plan_type": plan_type,
            "storage_path": storage_path_out or "",
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "progress_summary": f"{len(plan_progress)} sections completed",
        },
        "plan_progress": newly_completed,  # append reducer
        "compound_context": new_ctx,
    }


__all__ = ["ReadingPlanAgent", "ReadingPlanResult", "plan_node"]
