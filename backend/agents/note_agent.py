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
from storage.book_catalog import get_note_catalog
from utils.text import safe_id
from agents.obsidian_tools import TOOLS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM 提炼提示词
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """你是读书笔记整理助手。从用户与AI的对话中提炼关键信息。
只输出合法JSON，不要有多余文字或代码块标记。"""

_EXTRACT_TEMPLATE = """\
用户问题：{question}

AI回答：{answer}

请提炼以下信息并以JSON输出：
{{
  "question_summary": "用户疑惑的核心，一句话，15字以内",
  "answer_keypoints": ["回答要点1", "回答要点2"],
  "followup_questions": ["值得继续探究的问题1", "问题2"],
  "concepts": ["核心概念1", "概念2", "概念3"]
}}"""

_AGENT_SYSTEM_PROMPT = """你是一个具有极高自主性的卡片盒笔记法 (Zettelkasten) 专家。
当前用户正在阅读《{book_title}》。这是刚才的一轮问答：
【用户疑问】：{question}
【AI回答】：{answer}

系统已经为你初步提炼了以下要素：
- 疑问核心：{summary}
- 核心概念：{concepts}

【你的任务】
1. 利用 search_vault_for_concept 工具去 Obsidian 知识库中搜索上述“核心概念”，看看其他书籍是否也有相关笔记。
2. 整合问答内容和搜索到的跨书关联，撰写一段格式优美、带有 Obsidian 双向链接（如 [[其他书名]]）和标签（如 #概念）的 Markdown 笔记。
3. 必须调用 append_note_to_obsidian 工具，将写好的笔记保存到知识库中（注意传入 file 参数时直接使用书名，如 '{book_title}'）。
4. 完成保存后，结束任务并告诉用户你做了哪些知识串联。
"""

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
    升级版 NoteAgent：作为 ReAct Agent 运行，自主使用 Obsidian CLI 工具。
    1. LLM 提炼基本要素 (JSON)
    2. 触发 Tool-calling Agent 进行知识库搜索与编织
    3. Agent 自主调用 Obsidian CLI 写入 Markdown
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
        
        # 初始化 Obsidian ReAct Agent
        self._tools = TOOLS
        self._agent_executor = create_react_agent(
            self._llm,
            tools=self._tools
        )

    def process_qa(
        self, question: str, answer: str, book_title: str, book_id: str = ""
    ) -> NoteEntry | None:
        """deepread_book 后自动调用。提炼问答、触发 Agent 并入库。"""
        if not book_title:
            return None

        # 1. 提取结构化数据 (用于 VectorDB 和时间轴)
        try:
            extracted = self._extract(question, answer)
        except Exception as e:
            logger.warning("extraction failed: %s", e)
            return None

        entry = NoteEntry(
            book_title=book_title,
            date=datetime.now(tz=timezone.utc).isoformat(),
            **extracted,
        )

        # 2. 让 Agent 自主执行 Obsidian 搜索和写入
        try:
            logger.info("Starting Obsidian Note Agent for %s", book_title)
            
            # Format the system prompt with variables for this specific run
            # Since create_react_agent uses the modifier globally, we can just pass the formatted task in the human message
            task_msg = _AGENT_SYSTEM_PROMPT.format(
                book_title=book_title,
                question=question,
                answer=answer,
                summary=entry.question_summary,
                concepts=", ".join(entry.concepts)
            )
            
            self._agent_executor.invoke({
                "messages": [HumanMessage(content=task_msg + "\n\n请开始执行笔记整理任务。请务必最后调用 append_note_to_obsidian 工具进行保存。")]
            })
        except Exception as e:
            logger.error("Obsidian Agent execution failed: %s", e)

        # 3. 记录到目录，确保前端可以定位
        if book_id:
            note_path = self._resolve_note_path(book_id, book_title)
            get_note_catalog().upsert(book_id=book_id, file_path=str(note_path))

        logger.info("processed Q&A for 《%s》, concepts=%s", book_title, entry.concepts)
        return entry

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _resolve_note_path(self, book_id: str, book_title: str) -> Path:
        if book_id:
            record = get_note_catalog().get_by_book_id(book_id)
            if record:
                return Path(record["file_path"])
        # 改为使用书名，契合 Obsidian 的命名习惯
        safe = re.sub(r'[<>:"/\\|?*《》【】\r\n]', "_", book_title).strip("_. ") or "unknown"
        return self._notes_dir / f"{safe}.md"

    def _extract(self, question: str, answer: str) -> dict:
        msgs = [
            SystemMessage(content=_EXTRACT_SYSTEM),
            HumanMessage(content=_EXTRACT_TEMPLATE.format(
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
