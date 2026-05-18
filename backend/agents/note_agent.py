"""NoteAgent — post-processing hook, NOT a ReAct agent.

Triggered automatically after every deepread_book tool call.
Extracts structured insights from the Q&A and appends them to a per-book
Markdown file, while also indexing the entry in the ChromaDB notes collection
for cross-book semantic association.

CRUD operations (append_manual, get_note_content, list_books, get_timeline)
live in NoteService (services/note_service.py).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from config import get_settings
from llm.openai_client import get_llm
from prompts import get_prompts
from storage.book_catalog import get_note_catalog
from utils.text import safe_id


def _load_note_tools():
    """Load note tools (Notion backend only)."""
    _ = get_settings()  # keep settings initialization side effects
    from agents.notion_tools import TOOLS as _TOOLS
    logging.getLogger(__name__).info("NoteAgent backend=notion")
    return _TOOLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class NoteEntry:
    book_title: str
    date: str
    question_summary: str
    answer_keypoints: list[str]
    followup_questions: list[str]
    concepts: list[str]
    entry_type: str = "qa"
    cross_book_refs: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# NoteAgent
# ---------------------------------------------------------------------------

class NoteAgent:
    """
    升级版 NoteAgent：作为 ReAct Agent 运行，自主使用 Notion 工具。
    1. LLM 提炼基本要素 (JSON)
    2. 触发 Tool-calling Agent 进行知识库搜索与编织
    3. Agent 自主调用 Notion API 持久化笔记
    4. Upsert 到 ChromaDB notes 集合供前端时间轴展示
    """

    def __init__(
        self,
        *,
        notes_dir: Path | None = None,
        llm=None,
    ) -> None:
        settings = get_settings()
        self._notes_dir = Path(notes_dir or settings.note_storage_dir)
        self._notes_dir.mkdir(parents=True, exist_ok=True)
        self._llm = llm or get_llm(temperature=0.1)
        
        # 初始化 ReAct Agent（固定使用 Notion 后端）
        self._tools = _load_note_tools()
        self._agent_executor = create_react_agent(
            self._llm,
            tools=self._tools
        )

    def process_qa(
        self, question: str, answer: str, book_title: str, book_id: str = "",
        user_id: str = "default",
        *, locale: str | None = None,
    ) -> NoteEntry | None:
        """deepread_book 后自动调用。提炼问答、触发 Agent 并入库。"""
        if not book_title:
            return None

        prompts = get_prompts(locale).note
        note_tool_prompts = get_prompts(locale).note_tools

        # Update tool descriptions per-locale before handing tools to the ReAct agent.
        try:
            for t in self._tools:
                if t.name == "read_past_notes":
                    t.description = note_tool_prompts.read_past_desc
                elif t.name == "search_vault_for_concept":
                    t.description = note_tool_prompts.search_vault_desc
                elif t.name == "append_note_to_workspace":
                    t.description = note_tool_prompts.append_note_desc
        except Exception as e:
            logger.debug("update note tool desc failed: %s", e)

        # 1. 提取结构化数据 (用于 VectorDB 和时间轴)
        try:
            extracted = self._extract(question, answer, prompts=prompts)
        except Exception as e:
            logger.warning("extraction failed: %s", e)
            return None

        entry = NoteEntry(
            book_title=book_title,
            date=datetime.now(tz=timezone.utc).isoformat(),
            **extracted,
        )

        # 2. 让 Agent 自主执行笔记搜索和写入
        try:
            logger.info("Starting Note Agent for %s", book_title)

            task_msg = prompts.agent_system_template.format(
                book_title=book_title,
                question=question,
                answer=answer,
                summary=entry.question_summary,
                concepts=", ".join(entry.concepts),
            )

            self._agent_executor.invoke({
                "messages": [HumanMessage(content=task_msg + prompts.agent_task_suffix)]
            })
        except Exception as e:
            logger.error("Note Agent execution failed: %s", e)

        # 3. 记录到目录，确保前端可以定位
        if book_id:
            note_path = self._resolve_note_path(book_id, book_title)
            get_note_catalog().upsert(owner_user_id=user_id, book_id=book_id, file_path=str(note_path))

        logger.info("processed Q&A for 《%s》, concepts=%s", book_title, entry.concepts)
        return entry

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _resolve_note_path(self, book_id: str, book_title: str) -> Path:
        if book_id:
            # 保守兼容：路径解析阶段不强制 owner；如需严格 owner，请传入 user_id 并改签名
            record = None
            if record:
                return Path(record["file_path"])
        # 改为使用书名，便于识别
        safe = re.sub(r'[<>:"/\\|?*《》【】\r\n]', "_", book_title).strip("_. ") or "unknown"
        return self._notes_dir / f"{safe}.md"

    def _extract(self, question: str, answer: str, *, prompts=None) -> dict:
        prompts = prompts or get_prompts().note
        msgs = [
            SystemMessage(content=prompts.extract_system),
            HumanMessage(content=prompts.extract_template.format(
                question=question[:800],
                answer=answer[:1500],
            )),
        ]
        resp = self._llm.invoke(msgs)
        text = resp.content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        return {
            "question_summary": str(data.get("question_summary", ""))[:60],
            "answer_keypoints": [str(x) for x in data.get("answer_keypoints", [])],
            "followup_questions": [str(x) for x in data.get("followup_questions", [])],
            "concepts": [str(x) for x in data.get("concepts", [])],
        }


__all__ = ["NoteAgent", "NoteEntry"]
