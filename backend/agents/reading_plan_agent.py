"""ReadingPlanAgent — edit/extend only.

Plan creation (new) is handled by PlanGenerator + REST API.
This agent handles chat-initiated modification of existing plans.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import tool

from backend.llm.openai_client import get_llm, build_messages_context
from backend.rag.chroma.chroma_store import ChromaStore
from backend.rag.retriever import HybridConfig, HybridRetriever
from backend.storage.plan_storage import LocalPlanStorage, safe_plan_name
from backend.xai.citation import Citation, build_citations

_PLAN_SYSTEM = """你是"阅读计划助手"，专门帮用户修改或扩展已有的阅读计划。

工作流程：
1. 先调用 load_existing_plan 查看用户当前的计划
2. 如需了解章节结构，调用 get_chapter_structure
3. 根据用户要求修改或补充计划，输出完整的更新后计划

输出格式（Markdown，必须保持原有格式）：
- 保留 ## 章节进度 段落，更新复选框状态
- 保留 ## 建议日程 段落，按需更新内容
- 不要增加新的顶级标题
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
        plan_storage_dir: Path | None = None,
    ) -> None:
        from backend.config import get_settings
        settings = get_settings()

        self._store = store or ChromaStore()
        self._llm = llm or get_llm(temperature=0.4)
        self._k = k
        self._plan_dir = Path(plan_storage_dir or settings.plan_storage_dir)
        self._plan_dir.mkdir(parents=True, exist_ok=True)
        self._storage = LocalPlanStorage(root=self._plan_dir)

        self._retriever = HybridRetriever(
            store=self._store,
            collection_name=self._store.collection_name,
            config=HybridConfig(fetch_k=20, final_k=k),
            llm=self._llm,
        )

        self._current_docs: list[Document] = []
        self._current_book_title: str = ""
        self._react_agent = self._build_react_agent()

    def _build_react_agent(self):
        from langgraph.prebuilt import create_react_agent
        agent_self = self

        @tool
        def load_existing_plan() -> str:
            """加载当前书籍已有的阅读计划，修改前必须先调用此工具。"""
            path = agent_self._storage.find_by_book(agent_self._current_book_title)
            if not path:
                return "该书尚无阅读计划，请让用户先在 Reader Mode 中打开该书以自动生成计划。"
            try:
                return agent_self._storage.load(path)
            except Exception as e:
                return f"加载计划失败：{e}"

        @tool
        def get_chapter_structure(book_source: str) -> str:
            """获取指定书籍的章节结构（在需要了解章节顺序时调用）。"""
            try:
                docs = agent_self._store.get_all_documents(
                    collection_name=agent_self._store.collection_name,
                    filter={"source": book_source},
                )
            except Exception as e:
                return f"获取章节结构失败：{e}"

            if not docs:
                return f"书库中未找到《{book_source}》。"

            chapter_chars: dict[str, int] = {}
            for doc in docs:
                meta = doc.metadata or {}
                chapter = meta.get("chapter_title") or meta.get("section_title") or "未命名章节"
                chapter_chars[chapter] = chapter_chars.get(chapter, 0) + len(doc.page_content or "")

            lines = []
            for title, chars in chapter_chars.items():
                mins = chars / 300.0
                time_str = f"约{mins:.0f}分钟" if mins < 60 else f"约{mins/60:.1f}小时"
                lines.append(f"- {title}（{time_str}）")
            return "\n".join(lines)

        return create_react_agent(
            self._llm,
            [load_existing_plan, get_chapter_structure],
            prompt=_PLAN_SYSTEM,
        )

    def run(
        self,
        *,
        query: str,
        book_title: str,
        action: Literal["edit", "extend"] = "edit",
        memory_context: str = "",
        plan_messages: list[AnyMessage] | None = None,
    ) -> ReadingPlanResult:
        self._current_docs = []
        self._current_book_title = book_title

        verb = "修改" if action == "edit" else "扩展/补充"
        parts = [
            f"操作：{verb}《{book_title}》的阅读计划",
            f"用户要求：{query}",
        ]
        if memory_context:
            parts.append(f"[历史记录参考]\n{memory_context}")

        history = build_messages_context(plan_messages or [])
        input_messages = history + [("user", "\n\n".join(parts))]

        print(f"[ReadingPlanAgent] run book={book_title!r} action={action}", file=sys.stdout)

        result = self._react_agent.invoke(
            {"messages": input_messages},
            config={"recursion_limit": 12},
        )
        answer = result["messages"][-1].content

        # Persist updated plan
        storage_path = self._storage.find_by_book(book_title)
        try:
            if storage_path:
                self._storage.update(storage_path, answer)
            else:
                self._storage.save(answer, safe_plan_name(book_title))
        except Exception as e:
            print(f"[ReadingPlanAgent] storage failed: {e}", file=sys.stderr)

        return ReadingPlanResult(
            answer=answer,
            citations=build_citations(self._current_docs),
            retrieved_docs=list(self._current_docs),
        )


__all__ = ["ReadingPlanAgent", "ReadingPlanResult"]
