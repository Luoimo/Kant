# Notion Storage Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace local `.md` file storage for notes and plans with Notion Databases, while fixing two resilience issues: load() failures now degrade gracefully, and edit/extend updates the existing Notion page in-place instead of creating orphaned duplicates.

**Architecture:** Phase 1 hardens the existing storage Protocol and agent code (backend-agnostic). Phase 2 implements `NotionNoteStorage` / `NotionPlanStorage` and wires them in via a config-driven factory. No agent business logic changes.

**Tech Stack:** Python, `notion-client>=2.2.2` (official Notion SDK), `unittest.mock` for tests (no real API calls in tests).

---

## File Map

| Action | File | What changes |
|---|---|---|
| Modify | `backend/storage/note_storage.py` | `save() -> str \| None`, add `update()` to Protocol and `LocalNoteStorage` |
| Modify | `backend/storage/plan_storage.py` | Same changes for plan storage |
| Modify | `backend/storage/__init__.py` | Add `make_note_storage()` / `make_plan_storage()` factory functions |
| Modify | `backend/agents/note_agent.py` | `NoteAgent.run()`: try/except load(); `notes_node`: try/except save, call `update()` for edit/extend |
| Modify | `backend/agents/reading_plan_agent.py` | Same pattern for `ReadingPlanAgent.run()` and `plan_node` |
| Modify | `backend/config.py` | Add `notion_api_key`, `notion_notes_db_id`, `notion_plans_db_id` |
| Modify | `backend/agents/orchestrator_agent.py` | Replace `LocalNoteStorage`/`LocalPlanStorage` with factory calls at lines 425–426 |
| Create | `backend/storage/notion_storage.py` | `_markdown_to_blocks()`, `NotionNoteStorage`, `NotionPlanStorage` |
| Modify | `requirements.txt` | Add `notion-client>=2.2.2` |
| Modify | `tests/storage/test_storage.py` | Add `update()` tests for Local storage |
| Create | `tests/storage/test_notion_storage.py` | Unit tests for Notion storage (mocked client) |
| Modify | `tests/agents/test_note_agent.py` | Tests for load() degradation and update() in notes_node |
| Modify | `tests/agents/test_reading_plan_agent.py` | Same for plan_node |

---

## Task 1: Update Storage Protocols — `save()` return type + `update()` method

**Files:**
- Modify: `backend/storage/note_storage.py`
- Modify: `backend/storage/plan_storage.py`
- Modify: `tests/storage/test_storage.py`

- [ ] **Step 1: Write failing tests for `update()` and `save()` returning `None`**

Add to `tests/storage/test_storage.py` inside `TestLocalNoteStorage`:

```python
def test_update_overwrites_content(self, tmp_path):
    storage = LocalNoteStorage(root=tmp_path)
    path = storage.save("# Original", "note_001")
    storage.update(path, "# Updated")
    assert storage.load(path) == "# Updated"

def test_save_returns_str_or_none(self, tmp_path):
    storage = LocalNoteStorage(root=tmp_path)
    result = storage.save("content", "note_x")
    assert result is None or isinstance(result, str)
```

Add the same two tests inside `TestLocalPlanStorage` (replace `NoteStorage` → `PlanStorage`, `note_` → `plan_`).

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/storage/test_storage.py -v -k "update or str_or_none"
```
Expected: FAIL — `LocalNoteStorage has no attribute 'update'`

- [ ] **Step 3: Update `NoteStorage` Protocol in `backend/storage/note_storage.py`**

Change line 8 from:
```python
def save(self, content: str, note_id: str) -> str: ...
```
to:
```python
def save(self, content: str, note_id: str) -> str | None: ...
def update(self, storage_path: str, content: str) -> None: ...
```

Add `update()` to `_LocalMarkdownStorage`:
```python
def update(self, storage_path: str, content: str) -> None:
    Path(storage_path).write_text(content, encoding="utf-8")
```

- [ ] **Step 4: Update `PlanStorage` Protocol in `backend/storage/plan_storage.py`**

Same two changes: `save() -> str | None`, add `update()` to Protocol.
`LocalPlanStorage` inherits `update()` from `_LocalMarkdownStorage` — no extra code needed there.

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/storage/test_storage.py -v
```
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add backend/storage/note_storage.py backend/storage/plan_storage.py tests/storage/test_storage.py
git commit -m "feat: add update() to Storage protocols, relax save() return to str | None"
```

---

## Task 2: Harden Agent `load()` Path — Degrade to "new" on Failure

**Files:**
- Modify: `backend/agents/note_agent.py` (lines 92–105)
- Modify: `backend/agents/reading_plan_agent.py` (lines 129–144)
- Modify: `tests/agents/test_note_agent.py`
- Modify: `tests/agents/test_reading_plan_agent.py`

- [ ] **Step 1: Write failing test for `NoteAgent` load degradation**

Add to `tests/agents/test_note_agent.py`:

```python
class TestNoteAgentLoadDegradation:
    def test_load_failure_degrades_to_new(self, mock_store, mock_llm):
        """If load() raises, agent must not propagate the exception — degrades to 'new'."""
        failing_storage = MagicMock()
        failing_storage.load.side_effect = ConnectionError("Notion unavailable")

        agent = NoteAgent(store=mock_store, llm=mock_llm, note_storage=failing_storage)
        # Should not raise — must return a NoteResult even when load fails
        result = agent.run(
            query="修改笔记",
            action="edit",
            storage_path="some-page-id",
        )
        assert isinstance(result, NoteResult)
        assert result.answer  # some content was generated
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/agents/test_note_agent.py::TestNoteAgentLoadDegradation -v
```
Expected: FAIL — `ConnectionError: Notion unavailable` propagates

- [ ] **Step 3: Wrap `load()` in `NoteAgent.run()` with try/except**

In `backend/agents/note_agent.py`, replace lines 95–105:

```python
# 路径 1：edit/extend — 加载已有笔记（load 失败时降级为 new）
if action in ("edit", "extend") and storage_path and self.note_storage:
    try:
        existing_note = self.note_storage.load(storage_path)
        return self._modify_note(
            query=query,
            existing_note=existing_note,
            action=action,
            notes_format=notes_format,
            memory_context=memory_context,
            notes_messages=notes_messages,
        )
    except Exception as e:
        print(f"[NoteAgent] load failed, degrading to new: {e}", file=sys.stdout)
        action = "new"
```

- [ ] **Step 4: Write and apply symmetric fix for `ReadingPlanAgent`**

Add to `tests/agents/test_reading_plan_agent.py`:

```python
class TestReadingPlanAgentLoadDegradation:
    def test_load_failure_degrades_to_new(self, mock_store, mock_llm):
        failing_storage = MagicMock()
        failing_storage.load.side_effect = ConnectionError("Notion unavailable")

        agent = ReadingPlanAgent(store=mock_store, llm=mock_llm, plan_storage=failing_storage)
        result = agent.run(
            query="修改计划",
            action="edit",
            storage_path="some-page-id",
        )
        assert isinstance(result, ReadingPlanResult)
        assert result.answer
```

In `backend/agents/reading_plan_agent.py`, replace lines 133–144:

```python
# 路径 1：edit/extend — 加载已有计划并修改（load 失败时降级为 new）
if action in ("edit", "extend") and storage_path and self.plan_storage:
    try:
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
    except Exception as e:
        print(f"[ReadingPlanAgent] load failed, degrading to new: {e}", file=sys.stdout)
        action = "new"
```

- [ ] **Step 5: Run all agent tests**

```bash
pytest tests/agents/test_note_agent.py tests/agents/test_reading_plan_agent.py -v
```
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add backend/agents/note_agent.py backend/agents/reading_plan_agent.py tests/agents/test_note_agent.py tests/agents/test_reading_plan_agent.py
git commit -m "feat: degrade note/plan load() failure to new instead of propagating exception"
```

---

## Task 3: Harden Node Save/Update — Try/Except + In-Place Update for Edit/Extend

**Files:**
- Modify: `backend/agents/note_agent.py` (lines 249–256, `notes_node`)
- Modify: `backend/agents/reading_plan_agent.py` (lines 381–388, `plan_node`)
- Modify: `tests/agents/test_note_agent.py`
- Modify: `tests/agents/test_reading_plan_agent.py`

- [ ] **Step 1: Write failing tests for node save behavior**

Add to `tests/agents/test_note_agent.py`:

```python
class TestNotesNodeStorage:
    def _make_state(self, action="new", storage_path=""):
        return {
            "user_input": "整理笔记",
            "action": action,
            "notes_last_output": {"storage_path": storage_path},
            "notes_messages": [],
            "memory_context": "",
        }

    def test_new_action_calls_save(self, mock_store, mock_llm):
        storage = MagicMock()
        storage.save.return_value = "local/note_001.md"
        agent = NoteAgent(store=mock_store, llm=mock_llm, note_storage=storage)
        state = self._make_state(action="new")
        notes_node(state, agent=agent)
        storage.save.assert_called_once()
        storage.update.assert_not_called()

    def test_edit_action_calls_update_when_storage_path_exists(self, mock_store, mock_llm):
        storage = MagicMock()
        storage.load.return_value = "# Old Note"
        agent = NoteAgent(store=mock_store, llm=mock_llm, note_storage=storage)
        state = self._make_state(action="edit", storage_path="existing-page-id")
        notes_node(state, agent=agent)
        storage.update.assert_called_once_with("existing-page-id", mock_llm.invoke.return_value.content)
        storage.save.assert_not_called()

    def test_save_failure_does_not_raise(self, mock_store, mock_llm):
        storage = MagicMock()
        storage.save.side_effect = ConnectionError("Notion down")
        agent = NoteAgent(store=mock_store, llm=mock_llm, note_storage=storage)
        state = self._make_state(action="new")
        result = notes_node(state, agent=agent)
        assert result["notes_last_output"]["storage_path"] == ""
        assert result["answer"]  # answer is still returned
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/agents/test_note_agent.py::TestNotesNodeStorage -v
```
Expected: FAIL

- [ ] **Step 3: Update `notes_node` save block in `backend/agents/note_agent.py`**

Replace lines 249–256:

```python
# 持久化笔记（edit/extend 复用已有路径；new 创建新路径；失败静默降级）
storage_path_out: str | None = None
note_id = f"note_{thread_id}_{int(datetime.now(tz=timezone.utc).timestamp())}"
note_storage = getattr(deps, "note_storage", None) if deps else None
if not note_storage:
    note_storage = agent.note_storage
if note_storage:
    try:
        if action in ("edit", "extend") and storage_path:
            note_storage.update(storage_path, content)
            storage_path_out = storage_path
        else:
            storage_path_out = note_storage.save(content, note_id)
    except Exception as e:
        print(f"[notes_node] storage failed: {e}", file=sys.stdout)
        storage_path_out = None
```

- [ ] **Step 4: Apply symmetric fix to `plan_node` in `backend/agents/reading_plan_agent.py`**

Add tests to `tests/agents/test_reading_plan_agent.py` (same pattern as above, using `plan_node`, `action="edit"`, `plan_last_output`).

Replace lines 381–388 in `reading_plan_agent.py`:

```python
# 持久化计划（edit/extend 复用已有路径；new 创建新路径；失败静默降级）
storage_path_out: str | None = None
plan_id = f"plan_{thread_id}_{int(datetime.now(tz=timezone.utc).timestamp())}"
plan_storage = getattr(deps, "plan_storage", None) if deps else None
if not plan_storage:
    plan_storage = agent.plan_storage
if plan_storage:
    try:
        if action in ("edit", "extend") and storage_path:
            plan_storage.update(storage_path, content)
            storage_path_out = storage_path
        else:
            storage_path_out = plan_storage.save(content, plan_id)
    except Exception as e:
        print(f"[plan_node] storage failed: {e}", file=sys.stdout)
        storage_path_out = None
```

- [ ] **Step 5: Run all tests**

```bash
pytest tests/agents/ tests/storage/ -v
```
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add backend/agents/note_agent.py backend/agents/reading_plan_agent.py tests/agents/test_note_agent.py tests/agents/test_reading_plan_agent.py
git commit -m "feat: notes_node/plan_node — update in-place for edit/extend, catch save exceptions"
```

---

## Task 4: Config — Add Notion Fields

**Files:**
- Modify: `backend/config.py`
- Modify: `.env` (add commented-out example fields)

- [ ] **Step 1: Add three fields to `Settings` in `backend/config.py`**

After line 43 (`plan_storage_dir`), add:

```python
# Notion storage (optional; if set, notes and plans are stored in Notion Databases)
notion_api_key: str = ""
notion_notes_db_id: str = ""
notion_plans_db_id: str = ""
```

- [ ] **Step 2: Add example entries to `.env`**

Append to `.env`:

```env
# Notion storage (leave blank to use local file storage)
NOTION_API_KEY=
NOTION_NOTES_DB_ID=
NOTION_PLANS_DB_ID=
```

- [ ] **Step 3: Verify settings load without error**

```bash
python -c "from backend.config import get_settings; s = get_settings(); print(s.notion_api_key)"
```
Expected: prints empty string (no error)

- [ ] **Step 4: Commit**

```bash
git add backend/config.py .env
git commit -m "feat: add Notion config fields to Settings"
```

---

## Task 5: Markdown → Notion Blocks Converter

**Files:**
- Create: `backend/storage/notion_storage.py` (converter function only for now)
- Create: `tests/storage/test_notion_storage.py` (converter tests only)

- [ ] **Step 1: Write failing tests for the converter**

Create `tests/storage/test_notion_storage.py`:

```python
"""Tests for Notion storage — all Notion API calls are mocked."""
from __future__ import annotations
import pytest
from backend.storage.notion_storage import _markdown_to_blocks


class TestMarkdownToBlocks:
    def test_heading1(self):
        blocks = _markdown_to_blocks("# Title")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "heading_1"
        assert blocks[0]["heading_1"]["rich_text"][0]["text"]["content"] == "Title"

    def test_heading2(self):
        blocks = _markdown_to_blocks("## Section")
        assert blocks[0]["type"] == "heading_2"

    def test_heading3(self):
        blocks = _markdown_to_blocks("### Sub")
        assert blocks[0]["type"] == "heading_3"

    def test_bullet_dash(self):
        blocks = _markdown_to_blocks("- item one")
        assert blocks[0]["type"] == "bulleted_list_item"
        assert blocks[0]["bulleted_list_item"]["rich_text"][0]["text"]["content"] == "item one"

    def test_bullet_star(self):
        blocks = _markdown_to_blocks("* item two")
        assert blocks[0]["type"] == "bulleted_list_item"

    def test_paragraph(self):
        blocks = _markdown_to_blocks("plain text here")
        assert blocks[0]["type"] == "paragraph"
        assert blocks[0]["paragraph"]["rich_text"][0]["text"]["content"] == "plain text here"

    def test_blank_lines_skipped(self):
        blocks = _markdown_to_blocks("# H\n\n- item\n\n")
        assert len(blocks) == 2

    def test_mixed_content(self):
        md = "# Title\n## Section\n- bullet\nParagraph text"
        blocks = _markdown_to_blocks(md)
        types = [b["type"] for b in blocks]
        assert types == ["heading_1", "heading_2", "bulleted_list_item", "paragraph"]

    def test_raw_code_block_structure(self):
        """The raw Markdown is always stored as the last code block."""
        from backend.storage.notion_storage import _make_raw_code_block
        block = _make_raw_code_block("# hello")
        assert block["type"] == "code"
        assert block["code"]["rich_text"][0]["text"]["content"] == "# hello"
        assert block["code"]["language"] == "markdown"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/storage/test_notion_storage.py::TestMarkdownToBlocks -v
```
Expected: FAIL — `ModuleNotFoundError: notion_storage`

- [ ] **Step 3: Create `backend/storage/notion_storage.py` with converter only**

```python
"""Notion-backed storage implementations for notes and plans."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Markdown → Notion blocks (minimal custom converter)
# ---------------------------------------------------------------------------

def _rich_text(content: str) -> list[dict]:
    return [{"type": "text", "text": {"content": content}}]


def _markdown_to_blocks(text: str) -> list[dict]:
    """Convert Markdown text to a list of Notion block objects.

    Handles: H1/H2/H3, bullet lists (- and *), paragraphs.
    Blank lines are skipped. Bold markers (**) are left as plain text —
    round-trip fidelity relies on the raw code block, not rendered blocks.
    """
    blocks: list[dict] = []
    for line in text.splitlines():
        stripped = line.rstrip()
        if not stripped:
            continue
        if stripped.startswith("### "):
            blocks.append({"type": "heading_3", "heading_3": {"rich_text": _rich_text(stripped[4:])}})
        elif stripped.startswith("## "):
            blocks.append({"type": "heading_2", "heading_2": {"rich_text": _rich_text(stripped[3:])}})
        elif stripped.startswith("# "):
            blocks.append({"type": "heading_1", "heading_1": {"rich_text": _rich_text(stripped[2:])}})
        elif stripped.startswith("- ") or stripped.startswith("* "):
            blocks.append({"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": _rich_text(stripped[2:])}})
        else:
            blocks.append({"type": "paragraph", "paragraph": {"rich_text": _rich_text(stripped)}})
    return blocks


def _make_raw_code_block(raw_markdown: str) -> dict:
    """Return a Notion code block containing the raw Markdown string.

    This block is appended as the last block on every page, enabling
    lossless load() that doesn't depend on rendered block conversion.
    """
    return {
        "type": "code",
        "code": {
            "rich_text": _rich_text(raw_markdown),
            "language": "markdown",
        },
    }
```

- [ ] **Step 4: Run converter tests**

```bash
pytest tests/storage/test_notion_storage.py::TestMarkdownToBlocks -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add backend/storage/notion_storage.py tests/storage/test_notion_storage.py
git commit -m "feat: add Markdown-to-Notion-blocks converter with raw code block"
```

---

## Task 6: `NotionNoteStorage` and `NotionPlanStorage`

**Files:**
- Modify: `backend/storage/notion_storage.py` (add the two storage classes)
- Modify: `tests/storage/test_notion_storage.py` (add storage class tests)
- Modify: `requirements.txt`

- [ ] **Step 1: Add `notion-client` to `requirements.txt`**

Add after `mem0ai` line:
```
notion-client>=2.2.2
```

Install it:
```bash
pip install notion-client>=2.2.2
```

- [ ] **Step 2: Write failing tests for `NotionNoteStorage`**

Add to `tests/storage/test_notion_storage.py`:

```python
from unittest.mock import MagicMock, patch, call
from backend.storage.notion_storage import NotionNoteStorage, NotionPlanStorage


def _make_notion_settings(db_id="notes-db-id"):
    s = MagicMock()
    s.notion_api_key = "secret_test"
    s.notion_notes_db_id = db_id
    s.notion_plans_db_id = "plans-db-id"
    return s


class TestNotionNoteStorage:
    def _make_storage(self, mock_client):
        with patch("backend.storage.notion_storage.Client", return_value=mock_client):
            return NotionNoteStorage(_make_notion_settings())

    def test_save_creates_page_and_returns_page_id(self):
        client = MagicMock()
        client.pages.create.return_value = {"id": "page-uuid-001"}
        storage = self._make_storage(client)

        result = storage.save("# Hello", "note_001")

        assert result == "page-uuid-001"
        client.pages.create.assert_called_once()
        call_kwargs = client.pages.create.call_args[1]
        assert call_kwargs["parent"]["database_id"] == "notes-db-id"
        assert call_kwargs["properties"]["Title"]["title"][0]["text"]["content"] == "note_001"

    def test_save_returns_none_on_api_error(self):
        client = MagicMock()
        client.pages.create.side_effect = Exception("API error")
        storage = self._make_storage(client)

        result = storage.save("# Hello", "note_001")
        assert result is None

    def test_load_extracts_raw_markdown_from_code_block(self):
        client = MagicMock()
        client.blocks.children.list.return_value = {
            "results": [
                {"type": "heading_1", "heading_1": {"rich_text": [{"plain_text": "Title"}]}},
                {
                    "type": "code",
                    "code": {"rich_text": [{"plain_text": "# Title\n\n- bullet"}]},
                },
            ],
            "has_more": False,
        }
        storage = self._make_storage(client)

        result = storage.load("page-uuid-001")
        assert result == "# Title\n\n- bullet"

    def test_load_raises_on_api_error(self):
        client = MagicMock()
        client.blocks.children.list.side_effect = Exception("Not found")
        storage = self._make_storage(client)

        with pytest.raises(Exception, match="Not found"):
            storage.load("page-uuid-001")

    def test_update_deletes_old_blocks_and_appends_new(self):
        client = MagicMock()
        client.blocks.children.list.return_value = {
            "results": [{"id": "block-1"}, {"id": "block-2"}],
            "has_more": False,
        }
        storage = self._make_storage(client)
        storage.update("page-uuid-001", "# New Content")

        assert client.blocks.delete.call_count == 2
        client.blocks.delete.assert_any_call(block_id="block-1")
        client.blocks.delete.assert_any_call(block_id="block-2")
        client.blocks.children.append.assert_called_once()

    def test_list_queries_database_with_title_filter(self):
        client = MagicMock()
        client.databases.query.return_value = {
            "results": [{"id": "page-aaa"}, {"id": "page-bbb"}],
            "has_more": False,
        }
        storage = self._make_storage(client)

        result = storage.list(prefix="note_thread1")
        assert result == ["page-aaa", "page-bbb"]
        call_kwargs = client.databases.query.call_args[1]
        assert call_kwargs["filter"]["title"]["starts_with"] == "note_thread1"

    def test_delete_archives_page(self):
        client = MagicMock()
        storage = self._make_storage(client)
        storage.delete("page-uuid-001")
        client.pages.update.assert_called_once_with(page_id="page-uuid-001", archived=True)

    def test_init_raises_if_db_id_missing(self):
        s = _make_notion_settings(db_id="")
        with patch("backend.storage.notion_storage.Client"):
            with pytest.raises(ValueError, match="NOTION_NOTES_DB_ID"):
                NotionNoteStorage(s)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/storage/test_notion_storage.py::TestNotionNoteStorage -v
```
Expected: FAIL — `NotionNoteStorage` not defined yet

- [ ] **Step 4: Implement `NotionNoteStorage` in `backend/storage/notion_storage.py`**

Append after the converter functions:

```python
try:
    from notion_client import Client
except ImportError:
    Client = None  # type: ignore[assignment,misc]

from backend.storage.note_storage import NoteStorage
from backend.storage.plan_storage import PlanStorage


def _fetch_all_blocks(client: Any, page_id: str) -> list[dict]:
    """Fetch all blocks from a Notion page, following pagination cursors."""
    results: list[dict] = []
    response = client.blocks.children.list(block_id=page_id)
    results.extend(response["results"])
    while response.get("has_more"):
        response = client.blocks.children.list(
            block_id=page_id, start_cursor=response["next_cursor"]
        )
        results.extend(response["results"])
    return results


def _extract_raw_markdown(blocks: list[dict]) -> str | None:
    """Find the last code block and extract its plain_text (raw Markdown)."""
    for block in reversed(blocks):
        if block.get("type") == "code":
            rich_text = block["code"].get("rich_text", [])
            return "".join(rt.get("plain_text", "") for rt in rich_text)
    return None


def _blocks_to_plain_text(blocks: list[dict]) -> str:
    """Fallback: concatenate plain_text from all blocks (degraded quality)."""
    parts: list[str] = []
    for block in blocks:
        btype = block.get("type", "")
        content = block.get(btype, {}).get("rich_text", [])
        text = "".join(rt.get("plain_text", "") for rt in content)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _append_blocks_in_batches(client: Any, page_id: str, blocks: list[dict]) -> None:
    """Append blocks to a Notion page in batches of 100 (API limit)."""
    for i in range(0, len(blocks), 100):
        client.blocks.children.append(block_id=page_id, children=blocks[i:i + 100])


class NotionNoteStorage:
    """Stores notes as pages in a Notion Database."""

    def __init__(self, settings: Any) -> None:
        if not settings.notion_notes_db_id:
            raise ValueError("NOTION_NOTES_DB_ID must be set to use Notion storage")
        if Client is None:
            raise ImportError("notion-client is not installed. Run: pip install notion-client")
        self._client = Client(auth=settings.notion_api_key)
        self._db_id = settings.notion_notes_db_id

    def save(self, content: str, note_id: str) -> str | None:
        try:
            rendered = _markdown_to_blocks(content)
            raw_block = _make_raw_code_block(content)
            all_blocks = rendered + [raw_block]

            page = self._client.pages.create(
                parent={"database_id": self._db_id},
                properties={
                    "Title": {"title": [{"text": {"content": note_id}}]},
                },
                children=all_blocks[:100],
            )
            page_id: str = page["id"]

            if len(all_blocks) > 100:
                _append_blocks_in_batches(self._client, page_id, all_blocks[100:])

            return page_id
        except Exception as e:
            print(f"[NotionNoteStorage] save failed: {e}", file=sys.stdout)
            return None

    def load(self, storage_path: str) -> str:
        blocks = _fetch_all_blocks(self._client, storage_path)
        raw = _extract_raw_markdown(blocks)
        if raw is not None:
            return raw
        return _blocks_to_plain_text(blocks)

    def update(self, storage_path: str, content: str) -> None:
        blocks = _fetch_all_blocks(self._client, storage_path)
        for block in blocks:
            self._client.blocks.delete(block_id=block["id"])
        new_blocks = _markdown_to_blocks(content) + [_make_raw_code_block(content)]
        _append_blocks_in_batches(self._client, storage_path, new_blocks)

    def list(self, prefix: str = "") -> list[str]:
        filter_: dict = {}
        if prefix:
            filter_ = {"property": "Title", "title": {"starts_with": prefix}}
        response = self._client.databases.query(database_id=self._db_id, filter=filter_)
        page_ids = [p["id"] for p in response["results"]]
        while response.get("has_more"):
            response = self._client.databases.query(
                database_id=self._db_id,
                filter=filter_,
                start_cursor=response["next_cursor"],
            )
            page_ids.extend(p["id"] for p in response["results"])
        return page_ids

    def delete(self, storage_path: str) -> None:
        self._client.pages.update(page_id=storage_path, archived=True)


class NotionPlanStorage:
    """Stores reading plans as pages in a Notion Database. Identical pattern to NotionNoteStorage."""

    def __init__(self, settings: Any) -> None:
        if not settings.notion_plans_db_id:
            raise ValueError("NOTION_PLANS_DB_ID must be set to use Notion storage")
        if Client is None:
            raise ImportError("notion-client is not installed. Run: pip install notion-client")
        self._client = Client(auth=settings.notion_api_key)
        self._db_id = settings.notion_plans_db_id

    def save(self, content: str, plan_id: str) -> str | None:
        try:
            rendered = _markdown_to_blocks(content)
            raw_block = _make_raw_code_block(content)
            all_blocks = rendered + [raw_block]

            page = self._client.pages.create(
                parent={"database_id": self._db_id},
                properties={
                    "Title": {"title": [{"text": {"content": plan_id}}]},
                },
                children=all_blocks[:100],
            )
            page_id: str = page["id"]

            if len(all_blocks) > 100:
                _append_blocks_in_batches(self._client, page_id, all_blocks[100:])

            return page_id
        except Exception as e:
            print(f"[NotionPlanStorage] save failed: {e}", file=sys.stdout)
            return None

    def load(self, storage_path: str) -> str:
        blocks = _fetch_all_blocks(self._client, storage_path)
        raw = _extract_raw_markdown(blocks)
        if raw is not None:
            return raw
        return _blocks_to_plain_text(blocks)

    def update(self, storage_path: str, content: str) -> None:
        blocks = _fetch_all_blocks(self._client, storage_path)
        for block in blocks:
            self._client.blocks.delete(block_id=block["id"])
        new_blocks = _markdown_to_blocks(content) + [_make_raw_code_block(content)]
        _append_blocks_in_batches(self._client, storage_path, new_blocks)

    def list(self, prefix: str = "") -> list[str]:
        filter_: dict = {}
        if prefix:
            filter_ = {"property": "Title", "title": {"starts_with": prefix}}
        response = self._client.databases.query(database_id=self._db_id, filter=filter_)
        page_ids = [p["id"] for p in response["results"]]
        while response.get("has_more"):
            response = self._client.databases.query(
                database_id=self._db_id,
                filter=filter_,
                start_cursor=response["next_cursor"],
            )
            page_ids.extend(p["id"] for p in response["results"])
        return page_ids

    def delete(self, storage_path: str) -> None:
        self._client.pages.update(page_id=storage_path, archived=True)


__all__ = [
    "_markdown_to_blocks",
    "_make_raw_code_block",
    "NotionNoteStorage",
    "NotionPlanStorage",
]
```

- [ ] **Step 5: Add `TestNotionPlanStorage` to test file**

Add a minimal test class that mirrors `TestNotionNoteStorage` but uses `NotionPlanStorage` and `notion_plans_db_id`:

```python
class TestNotionPlanStorage:
    def _make_storage(self, mock_client):
        with patch("backend.storage.notion_storage.Client", return_value=mock_client):
            return NotionPlanStorage(_make_notion_settings())

    def test_save_creates_page_and_returns_page_id(self):
        client = MagicMock()
        client.pages.create.return_value = {"id": "page-plan-001"}
        storage = self._make_storage(client)
        result = storage.save("## Plan", "plan_001")
        assert result == "page-plan-001"

    def test_init_raises_if_db_id_missing(self):
        s = _make_notion_settings()
        s.notion_plans_db_id = ""
        with patch("backend.storage.notion_storage.Client"):
            with pytest.raises(ValueError, match="NOTION_PLANS_DB_ID"):
                NotionPlanStorage(s)
```

- [ ] **Step 6: Run all Notion storage tests**

```bash
pytest tests/storage/test_notion_storage.py -v
```
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add backend/storage/notion_storage.py tests/storage/test_notion_storage.py requirements.txt
git commit -m "feat: implement NotionNoteStorage and NotionPlanStorage"
```

---

## Task 7: Factory Functions + Orchestrator Wiring

**Files:**
- Modify: `backend/storage/__init__.py`
- Modify: `backend/agents/orchestrator_agent.py` (lines 425–426)

- [ ] **Step 1: Add factory functions to `backend/storage/__init__.py`**

Replace the entire file:

```python
from pathlib import Path

from .note_storage import NoteStorage, LocalNoteStorage
from .plan_storage import PlanStorage, LocalPlanStorage


def make_note_storage(settings) -> NoteStorage:
    """Return NotionNoteStorage if NOTION_API_KEY is set, else LocalNoteStorage."""
    if settings.notion_api_key:
        from .notion_storage import NotionNoteStorage
        return NotionNoteStorage(settings)
    return LocalNoteStorage(Path(settings.note_storage_dir))


def make_plan_storage(settings) -> PlanStorage:
    """Return NotionPlanStorage if NOTION_API_KEY is set, else LocalPlanStorage."""
    if settings.notion_api_key:
        from .notion_storage import NotionPlanStorage
        return NotionPlanStorage(settings)
    return LocalPlanStorage(Path(settings.plan_storage_dir))


__all__ = [
    "NoteStorage", "LocalNoteStorage",
    "PlanStorage", "LocalPlanStorage",
    "make_note_storage", "make_plan_storage",
]
```

- [ ] **Step 2: Update `orchestrator_agent.py` imports and call site**

Add to the import block at the top of `backend/agents/orchestrator_agent.py`:
```python
from backend.storage import make_note_storage, make_plan_storage
```

Replace lines 425–426:
```python
# Before
note_storage=LocalNoteStorage(Path(settings.note_storage_dir)),
plan_storage=LocalPlanStorage(Path(settings.plan_storage_dir)),

# After
note_storage=make_note_storage(settings),
plan_storage=make_plan_storage(settings),
```

- [ ] **Step 3: Verify factory routes correctly (no real Notion token needed)**

```bash
python -c "
from backend.config import get_settings
from backend.storage import make_note_storage
s = get_settings()
storage = make_note_storage(s)
print(type(storage).__name__)
"
```
Expected: prints `LocalNoteStorage` (since `NOTION_API_KEY` is empty in `.env`)

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add backend/storage/__init__.py backend/agents/orchestrator_agent.py
git commit -m "feat: wire Notion storage via config factory; orchestrator uses make_note/plan_storage()"
```

---

## Verification

After all tasks complete:

```bash
# Full test suite must pass
pytest tests/ -v

# Smoke test: server starts without errors (local mode, no Notion key)
uvicorn backend.main:app --reload
```

To test with Notion: set `NOTION_API_KEY`, `NOTION_NOTES_DB_ID`, `NOTION_PLANS_DB_ID` in `.env`, restart, send a chat request with `active_tab=notes`. Verify the Notion Database receives a new page.
