# DeepRead Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 4 agents + queue architecture with a single DeepRead ReAct agent that has one unified `search_books` tool covering single-book, cross-book, and chapter retrieval.

**Architecture:** Single `DeepReadAgent` with `search_books(search_query, scope, chapter)` tool. The `scope` parameter (`"current_book"` | `"all_books"`) controls filter logic inside the closure; the LLM decides which to use. System prompt covers 4 behavior modes: concept explanation, chapter summary, cross-book comparison, Socratic challenge. Plan infrastructure is removed entirely. NoteAgent auto-hook is preserved unchanged.

**Tech Stack:** FastAPI · LangGraph · LangChain tools · ChromaDB · HybridRetriever · pytest

---

## File Map

**Modify:**
- `backend/agents/deepread_agent.py` — replace 3 old tools with `search_books`; new system prompt
- `backend/api/chat.py` — remove `active_tab` field; update `_TOOL_STATUS`
- `backend/main.py` — remove `PlanEditor` import and `app.state.plan_editor`
- `backend/api/reader.py` — remove all plan endpoints (`/init`, `/plan`, `/progress`)
- `frontend/src/api/index.js` — remove `readerApi.init`, `.plan`, `.progress`

**Create:**
- `tests/agents/test_deepread_agent.py` — unit tests for `search_books` tool logic

**Delete:**
- `backend/agents/plan_editor.py`
- `tests/agents/test_plan_editor.py`
- `tests/agents/test_plan_generator.py`

**Unchanged:**
- `backend/agents/note_agent.py`
- `backend/rag/` (entire directory)
- `backend/storage/`
- `backend/memory/`
- `backend/security/`
- `backend/api/notes.py`
- `backend/api/books.py`

---

## Task 1: Add `_search_books_impl` + tests to DeepReadAgent

The testable core of the new tool. Extracted as a method so tests can call it directly without spinning up LangGraph.

**Files:**
- Modify: `backend/agents/deepread_agent.py`
- Create: `tests/agents/test_deepread_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/agents/test_deepread_agent.py
from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document

from backend.agents.deepread_agent import DeepReadAgent, DeepReadConfig


def _make_doc(content: str, source: str = "data/books/kant.epub", chapter: str = "第一章") -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "section_title": chapter, "book_title": "纯粹理性批判"},
    )


@pytest.fixture
def agent():
    mock_store = MagicMock()
    mock_llm = MagicMock()
    a = DeepReadAgent(store=mock_store, llm=mock_llm, config=DeepReadConfig(k=6, fetch_k=20))
    # Inject a mock retriever directly
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [_make_doc("先验统觉是…")]
    a._retriever = mock_retriever
    return a, mock_retriever


def test_search_books_current_book_applies_source_filter(agent):
    a, mock_retriever = agent
    a._search_books_impl(
        search_query="先验统觉",
        scope="current_book",
        book_source="data/books/kant.epub",
        chapter=None,
    )
    mock_retriever.search.assert_called_once_with(
        "先验统觉", filter={"source": "data/books/kant.epub"}
    )


def test_search_books_all_books_no_filter(agent):
    a, mock_retriever = agent
    a._search_books_impl(
        search_query="自由意志",
        scope="all_books",
        book_source="data/books/kant.epub",
        chapter=None,
    )
    mock_retriever.search.assert_called_once_with("自由意志", filter=None)


def test_search_books_chapter_prepends_to_query(agent):
    a, mock_retriever = agent
    a._search_books_impl(
        search_query="总结",
        scope="current_book",
        book_source="data/books/kant.epub",
        chapter="先验感性论",
    )
    call_args = mock_retriever.search.call_args
    assert "先验感性论" in call_args[0][0]  # chapter prepended to query


def test_search_books_no_book_source_current_book_no_filter(agent):
    a, mock_retriever = agent
    a._search_books_impl(
        search_query="什么是哲学",
        scope="current_book",
        book_source=None,
        chapter=None,
    )
    mock_retriever.search.assert_called_once_with("什么是哲学", filter=None)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/agents/test_deepread_agent.py -v
```

Expected: `AttributeError: 'DeepReadAgent' object has no attribute '_search_books_impl'`

- [ ] **Step 3: Add `_search_books_impl` to `DeepReadAgent`**

In `backend/agents/deepread_agent.py`, add this method to the `DeepReadAgent` class after `invalidate_retriever`:

```python
def _search_books_impl(
    self,
    search_query: str,
    scope: str,
    book_source: str | None,
    chapter: str | None,
) -> list[Document]:
    """Core retrieval logic for search_books tool. Extracted for testability."""
    filter_ = {"source": book_source} if (scope == "current_book" and book_source) else None
    effective_query = f"{chapter} {search_query}" if chapter else search_query
    return self._get_retriever().search(effective_query, filter=filter_)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/agents/test_deepread_agent.py -v
```

Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/agents/test_deepread_agent.py backend/agents/deepread_agent.py
git commit -m "feat: add _search_books_impl with scope/chapter retrieval logic"
```

---

## Task 2: Replace old tools with `search_books` in `_build()` + new system prompt

**Files:**
- Modify: `backend/agents/deepread_agent.py`

- [ ] **Step 1: Replace `_SYSTEM_PROMPT` constant**

Replace the entire `_SYSTEM_PROMPT` string (lines 36–55 of the current file):

```python
_SYSTEM_PROMPT = """\
你是"阅读助手"，帮助用户深度理解哲学和社科书籍。

工具说明：
- search_books(search_query, scope, chapter)
  - scope="current_book"（默认）：在当前书中检索原文证据
  - scope="all_books"：跨全书库检索，用于跨书对比
  - chapter 非空时：聚焦于指定章节（用于章节摘要）

工作模式：

【概念解释模式】触发：用户划选文字，或问"什么意思/解释/这个概念"
- 调用 search_books 检索该词上下文
- 输出三层：① 词义 ② 书中含义（引用原文）③ 哲学/思想史背景

【章节摘要模式】触发："总结这章/这节讲了什么/梳理一下"
- 调用 search_books(chapter=当前章节名) 检索章节内容
- 输出：核心论点 → 论证结构 → 关键术语

【跨书对比模式】触发：涉及多本书、"对比/比较"，或问题超出当前书范围
- 调用 search_books(scope="all_books") 检索
- 书库中没有的书，用通识知识补充，末尾注明「（来自通识知识，非书库原文）」
- 输出：结构化对比两书观点

【苏格拉底模式】触发："我觉得/我认为/对吗/考我/测试我"
- 验证模式：搜索书中反例或支撑证据，提出追问，不直接给答案
- 测验模式：从书中抽取核心命题，出一道开放题

【常规问答】其余书中内容问题
- 必须有 search_books 的证据支撑，不编造书中事实
- 输出：结构化回答 + 末尾「引用」小节（书名·章节）
"""
```

- [ ] **Step 2: Replace `_build()` method**

Replace the entire `_build` method (lines 109–206):

```python
def _build(self, *, book_source: str | None, book_id: str):
    """Build a react_agent with a single search_books tool closure. Returns (agent, current_docs)."""
    current_docs: list[Document] = []
    config = self.config

    @tool
    def search_books(
        search_query: str,
        scope: str = "current_book",
        chapter: str | None = None,
    ) -> str:
        """搜索书库内容。
        search_query: 检索关键词或问题。
        scope: "current_book"（默认，当前书）或 "all_books"（跨全书库，用于跨书对比）。
        chapter: 章节名称，非空时聚焦该章节内容（用于章节摘要）。
        """
        docs = self._search_books_impl(search_query, scope, book_source, chapter)

        logger.info(
            "search_books query=%r scope=%r chapter=%r hits=%d",
            search_query, scope, chapter, len(docs),
        )
        if not docs:
            return "未找到相关内容，请尝试换一种关键词。"

        seen = {d.page_content[:100] for d in current_docs}
        for d in docs:
            key = d.page_content[:100]
            if key not in seen:
                seen.add(key)
                current_docs.append(d)

        display = docs[: max(1, config.max_evidence)]
        blocks: list[str] = []
        for i, d in enumerate(display, 1):
            meta = d.metadata or {}
            title = meta.get("book_title") or "未知书名"
            location = meta.get("section_title") or meta.get("chapter_title") or ""
            blocks.append(
                f"[证据{i}] 书名：{title}  章节：{location}\n"
                + (d.page_content or "").strip()[:600]
            )
        return "\n\n".join(blocks)

    from langgraph.prebuilt import create_react_agent
    react_agent = create_react_agent(
        self.llm,
        [search_books],
        prompt=_SYSTEM_PROMPT,
    )
    return react_agent, current_docs
```

- [ ] **Step 3: Run existing tests to check nothing broke**

```
pytest tests/agents/test_deepread_agent.py tests/agents/test_note_agent.py -v
```

Expected: all PASS (note_agent tests are independent)

- [ ] **Step 4: Commit**

```bash
git add backend/agents/deepread_agent.py
git commit -m "feat: replace 3 old tools with unified search_books tool and new system prompt"
```

---

## Task 3: Remove plan infrastructure

**Files:**
- Delete: `backend/agents/plan_editor.py`
- Delete: `tests/agents/test_plan_editor.py`
- Delete: `tests/agents/test_plan_generator.py`
- Modify: `backend/main.py`
- Modify: `backend/api/reader.py`

- [ ] **Step 1: Remove `PlanEditor` from `main.py`**

In `backend/main.py`, remove lines that import and instantiate `PlanEditor`:

Remove this import line:
```python
from backend.agents.plan_editor import PlanEditor
```

Remove this instantiation line:
```python
app.state.plan_editor = PlanEditor()
```

The lifespan function should look like:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.agents.deepread_agent import DeepReadAgent
    from backend.agents.note_agent import NoteAgent
    from backend.memory.mem0_store import Mem0Store
    from backend.storage.book_catalog import get_conversation_storage
    from backend.storage.note_vector_store import make_note_vector_store
    from backend.config import get_settings

    settings = get_settings()
    note_vector_store = make_note_vector_store(settings)

    app.state.agent = DeepReadAgent()
    app.state.note_agent = NoteAgent(note_vector_store=note_vector_store)
    app.state.mem0 = Mem0Store()
    app.state.conv = get_conversation_storage()

    logging.getLogger(__name__).info("app started")
    yield
    logging.getLogger(__name__).info("app stopped")
```

- [ ] **Step 2: Remove plan endpoints from `reader.py`**

Replace the entire content of `backend/api/reader.py` with:

```python
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.storage.book_catalog import get_book_catalog

router = APIRouter(prefix="/reader", tags=["reader"])


def _resolve(book_id: str) -> dict:
    """Return book catalog entry, raise 404 if not found."""
    entry = get_book_catalog().get_by_id(book_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"书籍不存在：{book_id}")
    return entry


@router.post("/{book_id}/open")
def reader_open(book_id: str) -> dict:
    """Mark book as reading when user opens it."""
    book = _resolve(book_id)
    get_book_catalog().update_status(book_id, "reading")
    return {"book_id": book_id, "book_title": book["title"]}
```

- [ ] **Step 3: Delete plan files**

```bash
rm backend/agents/plan_editor.py
rm tests/agents/test_plan_editor.py
rm tests/agents/test_plan_generator.py
```

- [ ] **Step 4: Verify app starts without plan imports**

```
pytest tests/ -v --ignore=tests/agents/test_plan_editor.py --ignore=tests/agents/test_plan_generator.py -x
```

Expected: no import errors for `plan_editor`

- [ ] **Step 5: Commit**

```bash
git add backend/main.py backend/api/reader.py
git commit -m "refactor: remove PlanEditor and plan endpoints"
```

---

## Task 4: Simplify `chat.py`

Remove `active_tab` field and update tool status labels.

**Files:**
- Modify: `backend/api/chat.py`

- [ ] **Step 1: Update `_TOOL_STATUS` and `ChatRequest`**

Replace `_TOOL_STATUS`:
```python
_TOOL_STATUS: dict[str, str] = {
    "search_books": "正在检索书库…",
}
```

Remove `active_tab` from `ChatRequest`:
```python
class ChatRequest(BaseModel):
    query: str
    user_id: str = "default"
    book_id: str | None = None
    thread_id: str = "default"
    selected_text: str | None = None
    current_chapter: str | None = None
```

In the `chat` endpoint, change the final return to remove `active_tab`:
```python
return ChatResponse(
    answer=result.answer,
    citations=[c.__dict__ for c in result.citations],
    retrieved_docs_count=len(result.retrieved_docs),
    intent="deepread",
)
```

- [ ] **Step 2: Run chat-related tests**

```
pytest tests/ -k "chat" -v
```

Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add backend/api/chat.py
git commit -m "refactor: remove active_tab from chat API, update tool status labels"
```

---

## Task 5: Frontend cleanup

Remove unused plan API calls.

**Files:**
- Modify: `frontend/src/api/index.js`

- [ ] **Step 1: Remove plan methods from `readerApi`**

Replace `readerApi` in `frontend/src/api/index.js`:

```javascript
export const readerApi = {
  open: (bookId) => api.post(`/reader/${bookId}/open`),
}
```

- [ ] **Step 2: Verify no component imports the removed methods**

```bash
grep -r "readerApi.init\|readerApi.plan\|readerApi.progress" frontend/src/
```

Expected: no matches

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api/index.js
git commit -m "refactor: remove plan-related frontend API calls"
```

---

## Task 6: Full regression

- [ ] **Step 1: Run full test suite**

```
pytest tests/ -v
```

Expected: all tests PASS (plan-related test files are already deleted)

- [ ] **Step 2: Start the app and verify it boots**

```
uvicorn backend.main:app --reload
```

Expected: `app started` in logs, no import errors

- [ ] **Step 3: Smoke test streaming endpoint**

With the app running, send a test request:
```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是先验统觉", "book_id": null}'
```

Expected: SSE stream with `{"type":"thinking"}` → token events → `{"type":"done"}`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: complete deepread simplification - single agent, single tool"
```
