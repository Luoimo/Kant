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

from config import get_settings
from llm.openai_client import get_llm
from storage.book_catalog import get_note_catalog
from utils.text import safe_id

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
    轻量 hook，只做：
    1. LLM 提炼问答 → NoteEntry
    2. 跨书关联搜索（可选，依赖 NoteVectorStore）
    3. 追加格式化 Markdown 到 data/notes/{书名}.md
    4. Upsert 到 ChromaDB notes 集合

    用户手记 / 笔记读取 / 书目列表 → NoteService (services/note_service.py)
    """

    def __init__(
        self,
        *,
        notes_dir: Path | None = None,
        llm=None,
        note_vector_store=None,
    ) -> None:
        settings = get_settings()
        self._notes_dir = Path(notes_dir or settings.note_storage_dir)
        self._notes_dir.mkdir(parents=True, exist_ok=True)
        self._llm = llm or get_llm(temperature=0.1)
        self._note_vector_store = note_vector_store

    def process_qa(
        self, question: str, answer: str, book_title: str, book_id: str = ""
    ) -> NoteEntry | None:
        """deepread_book 后自动调用。提炼问答并写入文件 + 向量库。"""
        if not book_title:
            return None

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

        if self._note_vector_store:
            entry.cross_book_refs = self._find_associations(entry, book_title)

        note_path = self._resolve_note_path(book_id, book_title)
        self._append_to_file(entry, raw_question=question, path=note_path)
        if book_id:
            get_note_catalog().upsert(book_id=book_id, file_path=str(note_path))

        if self._note_vector_store:
            self._save_to_vector_store(entry)

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
            return self._notes_dir / f"{book_id}.md"
        # fallback for calls without book_id (e.g. tests)
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

    def _find_associations(self, entry: NoteEntry, exclude_book: str) -> list[dict]:
        search_text = entry.question_summary + " " + " ".join(entry.concepts)
        try:
            return self._note_vector_store.search_similar(
                text=search_text,
                exclude_book=exclude_book,
                top_k=2,
            )
        except Exception as e:
            logger.warning("association search failed: %s", e)
            return []

    def _append_to_file(self, entry: NoteEntry, raw_question: str, path: Path) -> None:
        dt = datetime.fromisoformat(entry.date)
        display_date = dt.strftime("%Y-%m-%d %H:%M")

        lines: list[str] = [
            "\n---\n",
            f"## {display_date} · {entry.question_summary}\n",
            f"**疑问**：{raw_question.strip()}\n",
        ]

        if entry.answer_keypoints:
            lines.append("\n**回答要点**")
            lines.extend(f"- {kp}" for kp in entry.answer_keypoints)

        if entry.followup_questions:
            lines.append("\n**延伸追问**")
            lines.extend(f"- {q}" for q in entry.followup_questions)

        if entry.cross_book_refs:
            lines.append("")
            for ref in entry.cross_book_refs:
                ref_book = ref.get("book_title", "")
                ref_summary = ref.get("question_summary", "")
                ref_date = ref.get("date", "")
                lines.append(f'💡 关联：《{ref_book}》中的\u201c{ref_summary}\u201d（{ref_date}）')

        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _save_to_vector_store(self, entry: NoteEntry) -> None:
        entry_id = f"qa_{safe_id(entry.book_title)}_{entry.date}"
        content = f"{entry.question_summary} {' '.join(entry.concepts)}"
        self._note_vector_store.add_entry(
            entry_id=entry_id,
            content=content,
            metadata={
                "book_title": entry.book_title,
                "date": entry.date,
                "question_summary": entry.question_summary,
                "concepts": json.dumps(entry.concepts, ensure_ascii=False),
                "entry_type": "qa",
            },
        )


__all__ = ["NoteAgent", "NoteEntry"]
