"""PlanEditor — create, edit, and extend reading plans.

- generate()  : non-ReAct, triggered by REST API when user opens a book.
                Extracts chapter structure from ChromaDB (Python), calls LLM
                only for the schedule section, writes plan to disk.
- run()       : ReAct agent, triggered by chat for edit/extend actions.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from backend.llm.openai_client import get_llm, build_messages_context
from backend.rag.chroma.chroma_store import ChromaStore
from backend.storage.plan_storage import LocalPlanStorage
from backend.storage.book_catalog import get_plan_catalog
from backend.xai.citation import Citation

_SCHEDULE_SYSTEM = """你是阅读计划助手。根据书籍章节列表生成一段简短的每日/每周阅读建议。
要求：友好务实，100字以内，不要重复列出章节，只给出节奏建议。"""

_SCHEDULE_TEMPLATE = """\
书名：《{book_title}》
阅读目标：{reading_goal}
章节总数：{chapter_count}，预计总时长：{total_hours}

请生成简短的建议日程。"""

_EDIT_SYSTEM = """你是"阅读计划助手"，专门帮用户修改或扩展已有的阅读计划。

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
class PlanEditResult:
    answer: str
    citations: list[Citation]
    retrieved_docs: list


class PlanEditor:
    def __init__(
        self,
        *,
        store: ChromaStore | None = None,
        llm=None,
        plan_storage_dir: Path | None = None,
    ) -> None:
        from backend.config import get_settings
        settings = get_settings()

        self._store = store or ChromaStore()
        self._llm = llm or get_llm(temperature=0.4)
        self._plan_dir = Path(plan_storage_dir or settings.plan_storage_dir)
        self._plan_dir.mkdir(parents=True, exist_ok=True)
        self._storage = LocalPlanStorage(root=self._plan_dir)
        self._current_book_id: str = ""
        self._react_agent = self._build_react_agent()

    # ------------------------------------------------------------------
    # Public: generate (non-ReAct, REST API path)
    # ------------------------------------------------------------------

    def generate(
        self,
        book_title: str,
        *,
        book_source: str | None = None,
        book_id: str = "",
        reading_goal: str = "通读全书",
    ) -> str:
        """Generate (or regenerate) a plan. Returns the plan markdown string."""
        chapters = self._extract_chapters(book_source)
        schedule = self._call_schedule_llm(book_title, reading_goal, chapters)
        content = self._build_plan(book_title, reading_goal, chapters, schedule)
        file_path = self._write(book_title, content, book_id=book_id)
        if book_id:
            get_plan_catalog().upsert(book_id=book_id, file_path=str(file_path), reading_goal=reading_goal)
        print(f"[PlanEditor] generated plan for 《{book_title}》", file=sys.stdout)
        return content

    # ------------------------------------------------------------------
    # Public: run (ReAct agent, chat path)
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        query: str,
        book_title: str,
        book_id: str = "",
        action: Literal["edit", "extend"] = "edit",
        memory_context: str = "",
        plan_messages: list[AnyMessage] | None = None,
    ) -> PlanEditResult:
        self._current_book_id = book_id

        verb = "修改" if action == "edit" else "扩展/补充"
        parts = [
            f"操作：{verb}《{book_title}》的阅读计划",
            f"用户要求：{query}",
        ]
        if memory_context:
            parts.append(f"[历史记录参考]\n{memory_context}")

        history = build_messages_context(plan_messages or [])
        input_messages = history + [("user", "\n\n".join(parts))]

        print(f"[PlanEditor] run book={book_title!r} action={action}", file=sys.stdout)

        result = self._react_agent.invoke(
            {"messages": input_messages},
            config={"recursion_limit": 12},
        )
        answer = result["messages"][-1].content

        try:
            record = get_plan_catalog().get_by_book_id(book_id) if book_id else None
            if record:
                self._storage.update(record["file_path"], answer)
                get_plan_catalog().touch(book_id)
            else:
                stem = book_id if book_id else re.sub(r'[<>:"/\\|?*《》【】\r\n]', "_", book_title).strip("_. ") or "unknown"
                file_path = self._storage.save(answer, stem)
                if book_id and file_path:
                    get_plan_catalog().upsert(book_id=book_id, file_path=file_path)
        except Exception as e:
            print(f"[PlanEditor] storage failed: {e}", file=sys.stderr)

        return PlanEditResult(answer=answer, citations=[], retrieved_docs=[])

    # ------------------------------------------------------------------
    # Internal: shared chapter extraction
    # ------------------------------------------------------------------

    def _extract_chapters(self, book_source: str | None) -> list[tuple[str, str]]:
        """Return [(chapter_title, time_str), ...] from ChromaDB, or empty list."""
        if not book_source:
            return []
        try:
            docs = self._store.get_all_documents(
                collection_name=self._store.collection_name,
                filter={"source": book_source},
            )
        except Exception as e:
            print(f"[PlanEditor] chapter extraction failed: {e}", file=sys.stderr)
            return []

        if not docs:
            return []

        chapter_chars: dict[str, int] = {}
        for doc in docs:
            meta = doc.metadata or {}
            chapter = meta.get("section_title") or meta.get("chapter_title") or "未命名章节"
            chapter_chars[chapter] = chapter_chars.get(chapter, 0) + len(doc.page_content or "")

        result: list[tuple[str, str]] = []
        for title, chars in chapter_chars.items():
            mins = chars / 300.0
            time_str = f"约{mins:.0f}分钟" if mins < 60 else f"约{mins/60:.1f}小时"
            result.append((title, time_str))
        return result

    # ------------------------------------------------------------------
    # Internal: generate helpers
    # ------------------------------------------------------------------

    def _call_schedule_llm(
        self,
        book_title: str,
        reading_goal: str,
        chapters: list[tuple[str, str]],
    ) -> str:
        total_mins = sum(
            float(t.replace("约", "").replace("分钟", "").replace("小时", "")) * (60 if "小时" in t else 1)
            for _, t in chapters
        ) if chapters else 60.0
        total_hours = f"约{total_mins/60:.1f}小时" if total_mins >= 60 else f"约{total_mins:.0f}分钟"

        msgs = [
            SystemMessage(content=_SCHEDULE_SYSTEM),
            HumanMessage(content=_SCHEDULE_TEMPLATE.format(
                book_title=book_title,
                reading_goal=reading_goal or "通读全书",
                chapter_count=len(chapters) if chapters else "未知",
                total_hours=total_hours,
            )),
        ]
        try:
            resp = self._llm.invoke(msgs)
            return resp.content.strip()
        except Exception as e:
            print(f"[PlanEditor] schedule LLM failed: {e}", file=sys.stderr)
            return "建议每天安排固定阅读时间，循序渐进完成计划。"

    def _build_plan(
        self,
        book_title: str,
        reading_goal: str,
        chapters: list[tuple[str, str]],
        schedule: str,
    ) -> str:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        header = (
            f"# 《{book_title}》阅读计划\n\n"
            f"**生成时间：** {date_str}  \n"
            f"**阅读目标：** {reading_goal or '通读全书'}\n\n"
        )

        if chapters:
            progress_lines = "\n".join(f"- [ ] {title}（{time_str}）" for title, time_str in chapters)
            progress = f"## 章节进度\n\n{progress_lines}\n\n"
        else:
            progress = "## 章节进度\n\n（书库中暂无该书章节信息）\n\n"

        return header + progress + f"## 建议日程\n\n{schedule}\n"

    def _write(self, book_title: str, content: str, book_id: str = "") -> Path:
        stem = book_id if book_id else (
            re.sub(r'[<>:"/\\|?*《》【】\r\n]', "_", book_title).strip("_. ") or "unknown"
        )
        path = self._plan_dir / f"{stem}.md"
        path.write_text(content, encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Internal: ReAct agent
    # ------------------------------------------------------------------

    def _build_react_agent(self):
        from langgraph.prebuilt import create_react_agent
        agent_self = self

        @tool
        def load_existing_plan() -> str:
            """加载当前书籍已有的阅读计划，修改前必须先调用此工具。"""
            record = get_plan_catalog().get_by_book_id(agent_self._current_book_id)
            if not record:
                return "该书尚无阅读计划，请让用户先在 Reader Mode 中打开该书以自动生成计划。"
            try:
                return agent_self._storage.load(record["file_path"])
            except Exception as e:
                return f"加载计划失败：{e}"

        @tool
        def get_chapter_structure(book_source: str) -> str:
            """获取指定书籍的章节结构（在需要了解章节顺序时调用）。"""
            chapters = agent_self._extract_chapters(book_source)
            if not chapters:
                return f"书库中未找到《{book_source}》。"
            return "\n".join(f"- {title}（{time_str}）" for title, time_str in chapters)

        return create_react_agent(
            self._llm,
            [load_existing_plan, get_chapter_structure],
            prompt=_EDIT_SYSTEM,
        )


__all__ = ["PlanEditor", "PlanEditResult"]
