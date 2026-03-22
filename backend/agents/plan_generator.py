"""PlanGenerator — non-ReAct, triggered by REST API when user opens a book.

Generates a structured Markdown reading plan from chapter structure + LLM,
saves to data/plans/{safe_book_title}.md (one plan per book, overwrite on re-generate).

Chapter checkboxes are built deterministically from ChromaDB data.
LLM only generates the recommended schedule section.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from backend.config import get_settings
from backend.llm.openai_client import get_llm
from backend.rag.chroma.chroma_store import ChromaStore
from backend.storage.plan_storage import safe_plan_name

_SCHEDULE_SYSTEM = """你是阅读计划助手。根据书籍章节列表生成一段简短的每日/每周阅读建议。
要求：友好务实，100字以内，不要重复列出章节，只给出节奏建议。"""

_SCHEDULE_TEMPLATE = """\
书名：《{book_title}》
阅读目标：{reading_goal}
章节总数：{chapter_count}，预计总时长：{total_hours}

请生成简短的建议日程。"""


class PlanGenerator:
    """
    轻量计划生成器（非 ReAct）。
    1. 从 ChromaDB 提取章节结构（Python 直接计算，不经 LLM）
    2. LLM 仅生成"建议日程"段落
    3. 组合写入 {plan_storage_dir}/{safe_title}.md（覆盖写）
    """

    def __init__(
        self,
        *,
        store: ChromaStore | None = None,
        llm=None,
        plan_storage_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        self._store = store or ChromaStore()
        self._llm = llm or get_llm(temperature=0.3)
        self._dir = Path(plan_storage_dir or settings.plan_storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        book_title: str,
        *,
        book_source: str | None = None,
        reading_goal: str = "通读全书",
    ) -> str:
        """Generate (or regenerate) a plan. Returns the plan markdown string."""
        chapters = self._extract_chapters(book_source)
        schedule = self._call_llm(book_title, reading_goal, chapters)
        content = self._build_plan(book_title, reading_goal, chapters, schedule)
        self._write(book_title, content)
        print(f"[PlanGenerator] generated plan for 《{book_title}》", file=sys.stdout)
        return content

    # ------------------------------------------------------------------
    # Internal
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
            print(f"[PlanGenerator] chapter extraction failed: {e}", file=sys.stderr)
            return []

        if not docs:
            return []

        chapter_chars: dict[str, int] = {}
        for doc in docs:
            meta = doc.metadata or {}
            chapter = meta.get("chapter_title") or meta.get("section_title") or "未命名章节"
            chapter_chars[chapter] = chapter_chars.get(chapter, 0) + len(doc.page_content or "")

        result: list[tuple[str, str]] = []
        for title, chars in chapter_chars.items():
            mins = chars / 300.0
            time_str = f"约{mins:.0f}分钟" if mins < 60 else f"约{mins/60:.1f}小时"
            result.append((title, time_str))
        return result

    def _call_llm(
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
            print(f"[PlanGenerator] LLM call failed: {e}", file=sys.stderr)
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

        schedule_section = f"## 建议日程\n\n{schedule}\n"
        return header + progress + schedule_section

    def _write(self, book_title: str, content: str) -> None:
        path = self._dir / f"{safe_plan_name(book_title)}.md"
        path.write_text(content, encoding="utf-8")


__all__ = ["PlanGenerator"]
