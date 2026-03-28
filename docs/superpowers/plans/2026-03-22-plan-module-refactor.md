# Plan Module Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple plan generation from chat — plans auto-generate when a user opens a book in Reader Mode (REST API), while chat only handles plan modification (edit/extend).

**Architecture:** A new lightweight `PlanGenerator` (non-ReAct, like NoteAgent) handles one-shot auto-generation triggered by `POST /reader/{book_title}/init`. The existing `ReadingPlanAgent` is stripped down to edit/extend only and wired to a new `modify_plan` supervisor tool. Plan files are named by sanitized book title (`data/plans/{safe_title}.md`), one plan per book.

**Tech Stack:** FastAPI, LangGraph, LangChain ReAct, ChromaDB, LocalPlanStorage (file-backed .md)

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `backend/agents/plan_generator.py` | Non-ReAct auto-generation from chapter structure |
| Modify | `backend/storage/plan_storage.py` | Add `find_by_book(book_title)` to `LocalPlanStorage` |
| Modify | `backend/agents/reading_plan_agent.py` | Strip to edit/extend only; remove plan_node, new action, list_available_books, search_book_content tools |
| Modify | `backend/agents/orchestrator_agent.py` | Replace `create_reading_plan` + `search_saved_plans` with `modify_plan`; clean `GraphState`, `RequestContext` |
| Modify | `backend/api/chat.py` | Add `POST /reader/{book_title}/init`, `POST /reader/{book_title}/progress`, `GET /reader/{book_title}/plan` |
| Create | `tests/agents/test_plan_generator.py` | Unit tests for PlanGenerator |
| Create | `tests/api/test_reader_plan_endpoints.py` | Unit tests for 3 new endpoints |
| Modify | `tests/agents/test_reading_plan_agent.py` | Remove new-action tests, update for simplified agent |
| Modify | `tests/agents/test_orchestrator_state.py` | Remove plan_last_output/plan_progress assertions |

---

## Plan File Format (locked in)

Every generated plan must use this structure so the progress endpoint can toggle checkboxes and the frontend can parse sidebar state:

```markdown
# 《{book_title}》阅读计划

**生成时间：** 2026-03-22
**阅读目标：** {reading_goal}

## 章节进度

- [ ] 第1章 先验感性论（约45分钟）
- [ ] 第2章 先验分析论（约90分钟）

## 建议日程

{LLM-generated weekly/daily schedule}
```

---

## Task 1: `LocalPlanStorage.find_by_book()` + centralised `safe_plan_name`

Plan files are named `{safe_title}.md`. `safe_plan_name` is exported from `plan_storage.py` and imported everywhere (chat.py, plan_generator.py, reading_plan_agent.py) — single source of truth, no duplication. `find_by_book` is also added to the `PlanStorage` Protocol.

**Files:**
- Modify: `backend/storage/plan_storage.py`
- Modify: `tests/storage/test_storage.py`

- [ ] **Step 1.1: Write the failing test**

Add to `tests/storage/test_storage.py`:

```python
from backend.storage.plan_storage import LocalPlanStorage, safe_plan_name

class TestSafePlanName:
    def test_strips_chinese_brackets(self):
        assert safe_plan_name("《纯粹理性批判》") == "纯粹理性批判"

    def test_strips_angle_brackets(self):
        assert safe_plan_name("<test>") == "test"

    def test_fallback_for_empty(self):
        assert safe_plan_name("") == "unknown"


class TestLocalPlanStorageFindByBook:
    def test_find_by_book_returns_path_when_exists(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        # save using sanitised name (as all callers must do)
        storage.save("plan content", safe_plan_name("纯粹理性批判"))
        path = storage.find_by_book("纯粹理性批判")
        assert path is not None
        assert path.endswith(".md")

    def test_find_by_book_returns_none_when_missing(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        assert storage.find_by_book("不存在的书") is None

    def test_find_by_book_sanitizes_title(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        # write using the same sanitisation as find_by_book will look for
        storage.save("content", safe_plan_name("《纯粹理性批判》"))
        path = storage.find_by_book("《纯粹理性批判》")
        assert path is not None
```

- [ ] **Step 1.2: Run to verify it fails**

```
pytest tests/storage/test_storage.py::TestLocalPlanStorageFindByBook tests/storage/test_storage.py::TestSafePlanName -v
```
Expected: `FAIL` — `safe_plan_name` and `find_by_book` not defined.

- [ ] **Step 1.3: Implement in `backend/storage/plan_storage.py`**

Full replacement of the file:

```python
from __future__ import annotations
import re as _re
from typing import Protocol, runtime_checkable

from backend.storage.note_storage import _LocalMarkdownStorage


def safe_plan_name(book_title: str) -> str:
    """Sanitise a book title into a safe filename stem (no extension).
    Single source of truth — import this everywhere instead of duplicating the regex.
    """
    safe = _re.sub(r'[<>:"/\\|?*《》【】\r\n]', "_", book_title).strip("_. ")
    return safe or "unknown"


@runtime_checkable
class PlanStorage(Protocol):
    def save(self, content: str, plan_id: str) -> str | None: ...
    def load(self, storage_path: str) -> str: ...
    def list(self, prefix: str = "") -> list[str]: ...
    def delete(self, storage_path: str) -> None: ...
    def update(self, storage_path: str, content: str) -> None: ...
    def search(self, query: str, top_k: int = 5) -> list[tuple[str, str]]: ...
    def find_by_book(self, book_title: str) -> str | None: ...


class LocalPlanStorage(_LocalMarkdownStorage):
    """Stores plans as .md files under a configurable root directory."""

    def save(self, content: str, plan_id: str) -> str:
        return self._save(content, plan_id)

    def find_by_book(self, book_title: str) -> str | None:
        """Return storage_path if a plan for this book exists, else None."""
        path = self.root / f"{safe_plan_name(book_title)}.md"
        return str(path) if path.exists() else None


__all__ = ["PlanStorage", "LocalPlanStorage", "safe_plan_name"]
```

- [ ] **Step 1.4: Run tests to verify pass**

```
pytest tests/storage/test_storage.py::TestLocalPlanStorageFindByBook -v
```
Expected: 3 PASS.

- [ ] **Step 1.5: Commit**

```bash
git add backend/storage/plan_storage.py tests/storage/test_storage.py
git commit -m "feat: add LocalPlanStorage.find_by_book() for book-title-based plan lookup"
```

---

## Task 2: `PlanGenerator` — auto-generation from chapter structure

Non-ReAct class (like NoteAgent). Called by the REST endpoint when a user opens a book for the first time.

**Files:**
- Create: `backend/agents/plan_generator.py`
- Create: `tests/agents/test_plan_generator.py`

- [ ] **Step 2.1: Write the failing tests**

Create `tests/agents/test_plan_generator.py`:

```python
"""Tests for PlanGenerator — no real API calls, no real ChromaStore."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from backend.agents.plan_generator import PlanGenerator


def _make_store(docs: list[Document] | None = None) -> MagicMock:
    store = MagicMock()
    store.collection_name = "test_col"
    store.get_all_documents.return_value = docs or [
        Document(
            page_content="先验感性论内容" * 100,
            metadata={"chapter_title": "先验感性论", "source": "kant.epub"},
        ),
        Document(
            page_content="先验分析论内容" * 200,
            metadata={"chapter_title": "先验分析论", "source": "kant.epub"},
        ),
    ]
    return store


def _make_llm(plan_text: str = "## 建议日程\n\n第1周读第1章") -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=plan_text)
    return llm


class TestPlanGenerator:
    def test_generate_returns_markdown_string(self):
        with tempfile.TemporaryDirectory() as d:
            gen = PlanGenerator(
                store=_make_store(),
                llm=_make_llm(),
                plan_storage_dir=Path(d),
            )
            content = gen.generate("纯粹理性批判", book_source="kant.epub")
        assert isinstance(content, str)
        assert "纯粹理性批判" in content

    def test_generate_includes_chapter_checkboxes(self):
        with tempfile.TemporaryDirectory() as d:
            gen = PlanGenerator(
                store=_make_store(),
                llm=_make_llm(),
                plan_storage_dir=Path(d),
            )
            content = gen.generate("纯粹理性批判", book_source="kant.epub")
        assert "- [ ]" in content
        assert "先验感性论" in content

    def test_generate_writes_plan_file(self):
        with tempfile.TemporaryDirectory() as d:
            gen = PlanGenerator(
                store=_make_store(),
                llm=_make_llm(),
                plan_storage_dir=Path(d),
            )
            gen.generate("纯粹理性批判", book_source="kant.epub")
            md_files = list(Path(d).glob("*.md"))
        assert len(md_files) == 1

    def test_generate_with_reading_goal(self):
        with tempfile.TemporaryDirectory() as d:
            gen = PlanGenerator(
                store=_make_store(),
                llm=_make_llm(),
                plan_storage_dir=Path(d),
            )
            content = gen.generate(
                "纯粹理性批判",
                book_source="kant.epub",
                reading_goal="重点研究先验感性论",
            )
        assert "重点研究先验感性论" in content

    def test_generate_skips_rag_when_book_not_in_library(self):
        store = _make_store(docs=[])  # empty → book not in library
        with tempfile.TemporaryDirectory() as d:
            gen = PlanGenerator(
                store=store,
                llm=_make_llm(),
                plan_storage_dir=Path(d),
            )
            content = gen.generate("不存在的书", book_source="missing.epub")
        # should still return a plan (LLM-knowledge fallback)
        assert isinstance(content, str)
        assert len(content) > 10

    def test_generate_idempotent_overwrites_existing(self):
        """Calling generate twice for same book overwrites the file (not appends)."""
        with tempfile.TemporaryDirectory() as d:
            gen = PlanGenerator(
                store=_make_store(),
                llm=_make_llm("## 建议日程\n\n第一次生成"),
                plan_storage_dir=Path(d),
            )
            gen.generate("纯粹理性批判", book_source="kant.epub")

            gen2 = PlanGenerator(
                store=_make_store(),
                llm=_make_llm("## 建议日程\n\n第二次生成"),
                plan_storage_dir=Path(d),
            )
            gen2.generate("纯粹理性批判", book_source="kant.epub")

            content = list(Path(d).glob("*.md"))[0].read_text(encoding="utf-8")
        assert "第二次生成" in content
        assert "第一次生成" not in content
```

- [ ] **Step 2.2: Run to verify tests fail**

```
pytest tests/agents/test_plan_generator.py -v
```
Expected: `ERROR` — module not found.

- [ ] **Step 2.3: Implement `PlanGenerator`**

Create `backend/agents/plan_generator.py`:

```python
"""PlanGenerator — non-ReAct, triggered by REST API when user opens a book.

Generates a structured Markdown reading plan from chapter structure + LLM,
saves to data/plans/{safe_book_title}.md (one plan per book, overwrite on re-generate).
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from backend.config import get_settings
from backend.llm.openai_client import get_llm
from backend.rag.chroma.chroma_store import ChromaStore
from backend.storage.plan_storage import LocalPlanStorage, safe_plan_name

_GENERATOR_SYSTEM ="""你是阅读计划生成助手。根据书籍章节结构生成一份实用的阅读计划。

输出必须包含以下两个部分（严格遵守格式，不要有其他顶级标题）：

## 章节进度

每个章节一行，格式：- [ ] {章节名}（约{N}分钟）
按书中顺序列出所有章节。

## 建议日程

根据总时长给出每日/每周建议，风格友好务实，不超过200字。"""

_GENERATOR_TEMPLATE = """\
书名：《{book_title}》
阅读目标：{reading_goal}

章节结构与时间估算：
{chapter_lines}

请生成阅读计划。"""


class PlanGenerator:
    """
    轻量计划生成器（非 ReAct）。
    1. 从 ChromaDB 提取章节结构和字数
    2. LLM 生成结构化 Markdown
    3. 写入 {plan_storage_dir}/{safe_title}.md（覆盖写）
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
        chapter_lines = self._extract_chapters(book_source)
        plan_body = self._call_llm(book_title, reading_goal, chapter_lines)
        content = self._build_plan(book_title, reading_goal, plan_body)
        self._write(book_title, content)
        print(f"[PlanGenerator] generated plan for 《{book_title}》", file=sys.stdout)
        return content

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_chapters(self, book_source: str | None) -> str:
        if not book_source:
            return "（未指定书源，将基于书名生成通用计划）"
        try:
            docs = self._store.get_all_documents(
                collection_name=self._store.collection_name,
                filter={"source": book_source},
            )
        except Exception as e:
            print(f"[PlanGenerator] chapter extraction failed: {e}", file=sys.stderr)
            return "（章节结构获取失败，将生成通用计划）"

        if not docs:
            return "（书库中未找到该书，将基于书名生成通用计划）"

        chapter_chars: dict[str, int] = {}
        for doc in docs:
            meta = doc.metadata or {}
            chapter = meta.get("chapter_title") or meta.get("section_title") or "未命名章节"
            chapter_chars[chapter] = chapter_chars.get(chapter, 0) + len(doc.page_content or "")

        lines: list[str] = []
        for title, chars in chapter_chars.items():
            mins = chars / 300.0
            time_str = f"约{mins:.0f}分钟" if mins < 60 else f"约{mins/60:.1f}小时"
            lines.append(f"- {title}（{time_str}）")
        return "\n".join(lines)

    def _call_llm(self, book_title: str, reading_goal: str, chapter_lines: str) -> str:
        msgs = [
            SystemMessage(content=_GENERATOR_SYSTEM),
            HumanMessage(content=_GENERATOR_TEMPLATE.format(
                book_title=book_title,
                reading_goal=reading_goal or "通读全书",
                chapter_lines=chapter_lines,
            )),
        ]
        try:
            resp = self._llm.invoke(msgs)
            return resp.content.strip()
        except Exception as e:
            print(f"[PlanGenerator] LLM call failed: {e}", file=sys.stderr)
            return "## 章节进度\n\n（计划生成失败，请重试）\n\n## 建议日程\n\n请稍后重试。"

    def _build_plan(self, book_title: str, reading_goal: str, plan_body: str) -> str:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        return (
            f"# 《{book_title}》阅读计划\n\n"
            f"**生成时间：** {date_str}  \n"
            f"**阅读目标：** {reading_goal or '通读全书'}\n\n"
            f"{plan_body}\n"
        )

    def _write(self, book_title: str, content: str) -> None:
        path = self._dir / f"{safe_plan_name(book_title)}.md"
        path.write_text(content, encoding="utf-8")


__all__ = ["PlanGenerator"]
```

- [ ] **Step 2.4: Run tests to verify pass**

```
pytest tests/agents/test_plan_generator.py -v
```
Expected: 6 PASS.

- [ ] **Step 2.5: Commit**

```bash
git add backend/agents/plan_generator.py tests/agents/test_plan_generator.py
git commit -m "feat: add PlanGenerator for auto plan creation on book open"
```

---

## Task 3: Reader Mode REST Endpoints

Three endpoints for the frontend Reader Mode:
- `POST /reader/{book_title}/init` — auto-generate plan (idempotent)
- `GET /reader/{book_title}/plan` — get plan for sidebar
- `POST /reader/{book_title}/progress` — mark a chapter complete

**Files:**
- Modify: `backend/api/chat.py`
- Create: `tests/api/test_reader_plan_endpoints.py`

- [ ] **Step 3.1: Write failing tests**

Create `tests/api/test_reader_plan_endpoints.py`:

```python
"""Tests for Reader Mode plan endpoints."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.chat import app

_SAMPLE_PLAN = """\
# 《纯粹理性批判》阅读计划

**生成时间：** 2026-03-22
**阅读目标：** 通读全书

## 章节进度

- [ ] 先验感性论（约45分钟）
- [ ] 先验分析论（约90分钟）

## 建议日程

每天阅读一小时，预计两周完成。
"""


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_plan_generator():
    gen = MagicMock()
    gen.generate.return_value = _SAMPLE_PLAN
    return gen


class TestReaderInitEndpoint:
    def test_init_creates_plan(self, client, mock_plan_generator):
        with patch("backend.api.chat.PlanGenerator", return_value=mock_plan_generator):
            resp = client.post(
                "/reader/纯粹理性批判/init",
                json={"book_source": "kant.epub", "reading_goal": "通读"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["book_title"] == "纯粹理性批判"
        assert "plan" in data
        assert "纯粹理性批判" in data["plan"]

    def test_init_reading_goal_optional(self, client, mock_plan_generator):
        with patch("backend.api.chat.PlanGenerator", return_value=mock_plan_generator):
            resp = client.post(
                "/reader/纯粹理性批判/init",
                json={"book_source": "kant.epub"},
            )
        assert resp.status_code == 200


class TestReaderGetPlanEndpoint:
    def test_get_plan_returns_content(self, client, tmp_path):
        plan_file = tmp_path / "纯粹理性批判.md"
        plan_file.write_text(_SAMPLE_PLAN, encoding="utf-8")
        with patch("backend.api.chat._plan_storage_dir", return_value=tmp_path):
            resp = client.get("/reader/纯粹理性批判/plan")
        assert resp.status_code == 200
        assert resp.json()["book_title"] == "纯粹理性批判"

    def test_get_plan_returns_empty_when_not_exists(self, client, tmp_path):
        with patch("backend.api.chat._plan_storage_dir", return_value=tmp_path):
            resp = client.get("/reader/不存在的书/plan")
        assert resp.status_code == 200
        assert resp.json()["plan"] == ""


class TestReaderProgressEndpoint:
    def test_progress_marks_chapter_done(self, client, tmp_path):
        plan_file = tmp_path / "纯粹理性批判.md"
        plan_file.write_text(_SAMPLE_PLAN, encoding="utf-8")
        with patch("backend.api.chat._plan_storage_dir", return_value=tmp_path):
            resp = client.post(
                "/reader/纯粹理性批判/progress",
                json={"chapter": "先验感性论"},
            )
        assert resp.status_code == 200
        updated = plan_file.read_text(encoding="utf-8")
        assert "- [x] 先验感性论" in updated
        assert "- [ ] 先验分析论" in updated  # unchanged

    def test_progress_404_when_no_plan(self, client, tmp_path):
        with patch("backend.api.chat._plan_storage_dir", return_value=tmp_path):
            resp = client.post(
                "/reader/不存在的书/progress",
                json={"chapter": "第1章"},
            )
        assert resp.status_code == 404
```

- [ ] **Step 3.2: Run to verify they fail**

```
pytest tests/api/test_reader_plan_endpoints.py -v
```
Expected: `ERROR` — endpoints not defined.

- [ ] **Step 3.3: Add endpoints to `backend/api/chat.py`**

Add imports at top of file:
```python
import re as _re
from backend.agents.plan_generator import PlanGenerator
from backend.storage.plan_storage import safe_plan_name as _safe_plan_name
```

Add helper and models after existing notes helpers:
```python
def _plan_storage_dir() -> Path:
    from backend.config import get_settings
    return Path(get_settings().plan_storage_dir)


class ReaderInitRequest(BaseModel):
    book_source: str | None = None
    reading_goal: str = ""


class ReaderProgressRequest(BaseModel):
    chapter: str
```

Add the three endpoints:
```python
# ---------------------------------------------------------------------------
# Reader Mode — Plan endpoints
# ---------------------------------------------------------------------------

@app.post("/reader/{book_title}/init")
def reader_init(book_title: str, req: ReaderInitRequest) -> dict:
    """Auto-generate a reading plan when user opens a book. Idempotent."""
    gen = PlanGenerator()
    plan = gen.generate(
        book_title,
        book_source=req.book_source,
        reading_goal=req.reading_goal,
    )
    return {"book_title": book_title, "plan": plan}


@app.get("/reader/{book_title}/plan")
def reader_get_plan(book_title: str) -> dict:
    """Return the current plan markdown for sidebar display."""
    path = _plan_storage_dir() / f"{_safe_plan_name(book_title)}.md"
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    return {"book_title": book_title, "plan": content}


@app.post("/reader/{book_title}/progress")
def reader_progress(book_title: str, req: ReaderProgressRequest) -> dict:
    """Mark a chapter as complete (toggles checkbox in plan file)."""
    path = _plan_storage_dir() / f"{_safe_plan_name(book_title)}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="该书暂无阅读计划，请先初始化")

    content = path.read_text(encoding="utf-8")
    chapter = _re.escape(req.chapter.strip())
    updated = _re.sub(
        rf"- \[ \] ({chapter}[^\n]*)",
        r"- [x] \1",
        content,
    )
    if updated == content:
        raise HTTPException(status_code=404, detail=f"未找到章节：{req.chapter}")
    path.write_text(updated, encoding="utf-8")
    return {"ok": True, "book_title": book_title, "chapter": req.chapter}
```

- [ ] **Step 3.4: Run tests to verify pass**

```
pytest tests/api/test_reader_plan_endpoints.py -v
```
Expected: 7 PASS.

- [ ] **Step 3.5: Commit**

```bash
git add backend/api/chat.py tests/api/test_reader_plan_endpoints.py
git commit -m "feat: add Reader Mode plan endpoints (init, get, progress)"
```

---

## Task 4: Simplify `ReadingPlanAgent` to edit/extend only

Remove: `action="new"`, `list_available_books` tool, `search_book_content` tool, `plan_node` function. Add: plan lookup by book_title via `LocalPlanStorage.find_by_book`.

**Files:**
- Modify: `backend/agents/reading_plan_agent.py`
- Modify: `tests/agents/test_reading_plan_agent.py`

- [ ] **Step 4.1: Write updated tests first**

Replace `tests/agents/test_reading_plan_agent.py` content with:

```python
"""Tests for ReadingPlanAgent — edit/extend only, no new-plan creation."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from backend.agents.reading_plan_agent import ReadingPlanAgent, ReadingPlanResult

_FAKE_ANSWER = "## 更新后的计划\n\n每天读两章"
_EXISTING_PLAN = "# 《康德》阅读计划\n\n## 章节进度\n\n- [ ] 第1章（约30分钟）"


def _make_store() -> MagicMock:
    store = MagicMock()
    store.collection_name = "test_col"
    store.list_sources.return_value = ["kant.epub"]
    store.get_all_documents.return_value = [
        Document(page_content="内容" * 100, metadata={"chapter_title": "第1章", "source": "kant.epub"})
    ]
    return store


def _make_llm() -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=_FAKE_ANSWER)
    llm.bind_tools.return_value = llm
    return llm


def _make_react_graph(answer: str = _FAKE_ANSWER) -> MagicMock:
    graph = MagicMock()
    graph.invoke.return_value = {"messages": [AIMessage(content=answer)]}
    return graph


class TestReadingPlanAgentEdit:
    def test_run_edit_returns_result(self, tmp_path):
        # Write existing plan file
        plan_file = tmp_path / "康德.md"
        plan_file.write_text(_EXISTING_PLAN, encoding="utf-8")

        with patch("langgraph.prebuilt.create_react_agent", return_value=_make_react_graph()):
            agent = ReadingPlanAgent(
                store=_make_store(),
                llm=_make_llm(),
                plan_storage_dir=tmp_path,
            )
            result = agent.run(
                query="把计划改成每天两章",
                book_title="康德",
                action="edit",
            )

        assert isinstance(result, ReadingPlanResult)
        assert result.answer == _FAKE_ANSWER

    def test_run_extend_returns_result(self, tmp_path):
        plan_file = tmp_path / "康德.md"
        plan_file.write_text(_EXISTING_PLAN, encoding="utf-8")

        with patch("langgraph.prebuilt.create_react_agent", return_value=_make_react_graph()):
            agent = ReadingPlanAgent(
                store=_make_store(),
                llm=_make_llm(),
                plan_storage_dir=tmp_path,
            )
            result = agent.run(
                query="增加第二部分的内容",
                book_title="康德",
                action="extend",
            )

        assert isinstance(result, ReadingPlanResult)
        assert result.answer

    def test_run_saves_updated_plan(self, tmp_path):
        plan_file = tmp_path / "康德.md"
        plan_file.write_text(_EXISTING_PLAN, encoding="utf-8")

        with patch("langgraph.prebuilt.create_react_agent", return_value=_make_react_graph()):
            agent = ReadingPlanAgent(
                store=_make_store(),
                llm=_make_llm(),
                plan_storage_dir=tmp_path,
            )
            agent.run(query="修改计划", book_title="康德", action="edit")

        updated = plan_file.read_text(encoding="utf-8")
        assert "更新后的计划" in updated

    def test_run_no_existing_plan_still_returns_result(self, tmp_path):
        """If no plan exists yet, agent gracefully continues without load_existing_plan."""
        with patch("langgraph.prebuilt.create_react_agent", return_value=_make_react_graph()):
            agent = ReadingPlanAgent(
                store=_make_store(),
                llm=_make_llm(),
                plan_storage_dir=tmp_path,
            )
            result = agent.run(
                query="修改计划",
                book_title="不存在的书",
                action="edit",
            )
        assert isinstance(result, ReadingPlanResult)

    def test_plan_node_not_exported(self):
        """plan_node has been removed — only ReadingPlanAgent and ReadingPlanResult exported."""
        import backend.agents.reading_plan_agent as m
        assert not hasattr(m, "plan_node")
```

- [ ] **Step 4.2: Run to see current failures**

```
pytest tests/agents/test_reading_plan_agent.py -v
```
Note which pass and which fail — expect `plan_storage_dir` param and `plan_node` tests to fail.

- [ ] **Step 4.3: Rewrite `reading_plan_agent.py`**

Replace the entire file with:

```python
"""ReadingPlanAgent — edit/extend only.

Plan creation (new) is handled by PlanGenerator + REST API.
This agent handles chat-initiated modification of existing plans.
"""
from __future__ import annotations

import re
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

_PLAN_SYSTEM ="""你是"阅读计划助手"，专门帮用户修改或扩展已有的阅读计划。

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
```

- [ ] **Step 4.4: Run tests**

```
pytest tests/agents/test_reading_plan_agent.py -v
```
Expected: all PASS.

- [ ] **Step 4.5: Commit**

```bash
git add backend/agents/reading_plan_agent.py tests/agents/test_reading_plan_agent.py
git commit -m "refactor: simplify ReadingPlanAgent to edit/extend only, remove plan_node"
```

---

## Task 5: Update Orchestrator

Replace `create_reading_plan` + `search_saved_plans` with a single `modify_plan` tool. Clean up `GraphState` (remove `plan_last_output`, `plan_progress`). Update `RequestContext`. Update `GraphDeps` to hold `ReadingPlanAgent`.

**Files:**
- Modify: `backend/agents/orchestrator_agent.py`
- Modify: `tests/agents/test_orchestrator_state.py`

- [ ] **Step 5.1: Update orchestrator state tests first**

In `tests/agents/test_orchestrator_state.py`:

**a) Update the import block at the top — remove `PlanOutputMeta`:**
```python
# Before:
from backend.agents.orchestrator_agent import (
    GraphState,
    RequestContext,
    PlanOutputMeta,
    _has_compound_signals,
)

# After:
from backend.agents.orchestrator_agent import (
    GraphState,
    RequestContext,
    _has_compound_signals,
)
```

**b) Delete the entire `test_plan_output_meta_typeddict` function.**

**c) Replace `test_graphstate_has_storage_pointer_fields`:**
```python
def test_graphstate_has_storage_pointer_fields():
    # plan_last_output and plan_progress removed — plan is now managed by REST API
    annotations = GraphState.__annotations__
    assert "plan_last_output" not in annotations
    assert "plan_progress" not in annotations
```

**d) Replace `test_request_context_defaults`:**
```python
def test_request_context_defaults():
    ctx = RequestContext()
    assert ctx.citations == []
    assert ctx.retrieved_docs_count == 0
    assert ctx.intent == "deepread"
    assert not hasattr(ctx, "plan_last_output")
    assert not hasattr(ctx, "plan_progress")
```

- [ ] **Step 5.2: Run state tests to see failures**

```
pytest tests/agents/test_orchestrator_state.py -v
```
Expected: state tests fail because fields still exist.

- [ ] **Step 5.3: Update orchestrator**

Key changes in `backend/agents/orchestrator_agent.py`:

**a) Remove `PlanOutputMeta` TypedDict entirely.**

**b) Remove from `GraphState`:**
```python
# Remove these two fields:
plan_last_output: PlanOutputMeta | None
plan_progress: Annotated[list[str], lambda a, b: a + b]
```

**c) Remove from `RequestContext`:**
```python
# Remove these two fields:
plan_last_output: dict = field(default_factory=dict)
plan_progress: list[str] = field(default_factory=list)
```

**d) Update `GraphDeps` — remove `plan_storage`, update `plan_agent` type hint comment:**
```python
@dataclass
class GraphDeps:
    deepread_agent: DeepReadAgent
    notes_agent: NoteAgent
    plan_agent: ReadingPlanAgent   # edit/extend only
    recommend_agent: RecommendationAgent
    mem0: Mem0Store | None = None
```

**e) Replace the two tools in `_build_supervisor_tools`:**

Remove `create_reading_plan` and `search_saved_plans`. Add:

```python
@tool
def modify_plan(query: str, book_title: str, action: str = "edit") -> str:
    """修改或扩展用户已有的阅读计划。
    在用户要求调整计划内容、更新进度或扩展计划时调用。
    action: 'edit'（修改内容）或 'extend'（增加内容）
    book_title: 要修改计划的书名
    """
    from typing import Literal
    safe_action: Literal["edit", "extend"] = "extend" if action == "extend" else "edit"
    result = deps.plan_agent.run(
        query=query,
        book_title=book_title,
        action=safe_action,
        memory_context=memory_context,
    )
    ctx.intent = "plan"
    ctx.retrieved_docs_count += len(result.retrieved_docs)
    print("[Supervisor.tool] modify_plan done", file=sys.stdout)
    return result.answer
```

Return `[deepread_book, modify_plan, recommend_books]` (3 tools).

**f) Update `SUPERVISOR_SYSTEM` prompt:**
```python
SUPERVISOR_SYSTEM = """你是智能阅读助手的总协调器，可使用以下工具：

- deepread_book: 对书库中的书籍进行深度精读、解析和问答（每次调用后系统会自动记录笔记）
- modify_plan: 修改或扩展用户已有的阅读计划（在 Reader Mode 中打开书后计划会自动生成）
- recommend_books: 从整个出版世界中推荐值得精读的书籍

工作原则：
1. 分析用户请求，选择合适的工具（可依次调用多个）
2. 修改计划时：直接调用 modify_plan，传入书名和修改要求
3. 将所有工具结果整合为一份连贯完整的 Markdown 回答
4. 如检测到用户报告阅读进度（"XX章节我读完了"），调用 modify_plan 更新计划

重要规则：
- 只使用工具返回的内容作答，不编造不存在的事实
"""
```

**g) Update `_TAB_HINT` in `_react_supervisor` — rename the plan entry:**
```python
# Before:
_TAB_HINT = {
    "deepread": "deepread_book",
    "plan": "create_reading_plan",
    "recommend": "recommend_books",
}

# After:
_TAB_HINT = {
    "deepread": "deepread_book",
    "plan": "modify_plan",
    "recommend": "recommend_books",
}
```

**i) Update `build_minimal_supervisor_graph` — remove `plan_storage`, update `ReadingPlanAgent` construction:**

```python
deps = GraphDeps(
    deepread_agent=DeepReadAgent(store=store),
    notes_agent=NoteAgent(note_vector_store=note_vector_store),
    plan_agent=ReadingPlanAgent(store=store),
    recommend_agent=RecommendationAgent(store=store),
    mem0=mem0,
)
```

**j) Remove `make_plan_storage` import** (no longer used in orchestrator):
```python
# Remove this import:
from backend.storage import make_plan_storage
```

**k) Remove `plan_node` import from `reading_plan_agent`**:
```python
# Change:
from backend.agents.reading_plan_agent import ReadingPlanAgent, plan_node
# To:
from backend.agents.reading_plan_agent import ReadingPlanAgent
```

**l) Clean up `run_minimal_graph` init dict** — remove `plan_progress` and `plan_last_output` from the per-turn reset block:
```python
# Remove these keys from the init state dict in run_minimal_graph:
# "plan_last_output": None,
# "plan_progress": [],
```

- [ ] **Step 5.4: Run orchestrator state tests**

```
pytest tests/agents/test_orchestrator_state.py -v
```
Expected: all PASS.

- [ ] **Step 5.5: Run full test suite**

```
pytest tests/ -q
```
Fix any remaining failures before committing.

- [ ] **Step 5.6: Commit**

```bash
git add backend/agents/orchestrator_agent.py tests/agents/test_orchestrator_state.py
git commit -m "refactor: replace create_reading_plan+search_saved_plans with modify_plan tool"
```

---

## Task 6: Final verification

- [ ] **Step 6.1: Run full test suite**

```
pytest tests/ -v
```
Expected: all pass, no import errors.

- [ ] **Step 6.2: Verify no leftover references to removed symbols**

```bash
grep -r "plan_node\|plan_last_output\|plan_progress\|search_saved_plans\|create_reading_plan\|make_plan_storage\|PlanOutputMeta" backend/ tests/
```
Expected: zero matches. If `plan_progress` appears in `run_minimal_graph`'s init dict, remove it (step **l** from Task 5 covers this).

- [ ] **Step 6.3: Final commit**

```bash
git add -A
git commit -m "chore: verify plan refactor complete, all tests passing"
```

---

## Summary of Changes

| What changed | Why |
|---|---|
| `PlanGenerator` (new) | One-shot auto-generation on book open |
| `ReadingPlanAgent` | Stripped to edit/extend only, 2 tools instead of 4 |
| `plan_node` | Removed — plan no longer a graph node |
| `PlanOutputMeta` | Removed — no longer in chat state |
| `plan_last_output`, `plan_progress` in `GraphState` | Removed — plan state lives in files, not graph |
| Supervisor tools: 4 → 3 | `create_reading_plan` + `search_saved_plans` → `modify_plan` |
| New endpoints | `/reader/{book_title}/init`, `/plan`, `/progress` |
