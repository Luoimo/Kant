# Orchestrator Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove book-related parameters from the three supervisor tool signatures and inject them via closure so the LLM never has to infer which book it's operating on.

**Architecture:** A new `_resolve_book_title` helper looks up the human-readable title from ChromaDB metadata at the start of each `_react_supervisor` call. Both `book_source` (file path) and `book_title` (display name) are passed as additional parameters to `_build_supervisor_tools`, where they are closed over into the tool functions. The three tools lose their book-related parameters entirely.

**Tech Stack:** Python, LangGraph `create_react_agent`, LangChain `@tool`, ChromaStore metadata API.

**Spec:** `docs/superpowers/specs/2026-03-22-orchestrator-simplification-design.md`

---

## File Map

| File | Action | What changes |
|---|---|---|
| `backend/agents/orchestrator_agent.py` | Modify | `_resolve_book_title`, `_title_from_docs` helpers added; `_build_supervisor_tools` signature + all 3 tool bodies changed; `_react_supervisor` extracts and passes book values; `SUPERVISOR_SYSTEM` rule 2 updated |
| `tests/agents/test_orchestrator_state.py` | Modify | New unit tests for tool closure injection and None-case behavior |

No other files change.

---

### Task 1: Add `_resolve_book_title` and `_title_from_docs` helpers

**Files:**
- Modify: `backend/agents/orchestrator_agent.py`
- Test: `tests/agents/test_orchestrator_state.py`

- [ ] **Step 1: Write failing tests for both helpers**

Augment the **existing** import block at the top of `tests/agents/test_orchestrator_state.py`:

```python
from backend.agents.orchestrator_agent import (
    GraphState,
    RequestContext,
    _resolve_book_title,   # add this line
    _title_from_docs,      # add this line
)
from unittest.mock import MagicMock   # add this line
```

Then add the following tests to the bottom of the file:


def test_resolve_book_title_returns_matching_title():
    store = MagicMock()
    store.list_book_titles.return_value = [
        {"source": "data/books/kant.epub", "book_title": "纯粹理性批判"},
        {"source": "data/books/hegel.epub", "book_title": "精神现象学"},
    ]
    assert _resolve_book_title("data/books/kant.epub", store) == "纯粹理性批判"


def test_resolve_book_title_returns_empty_when_not_found():
    store = MagicMock()
    store.list_book_titles.return_value = [
        {"source": "data/books/hegel.epub", "book_title": "精神现象学"},
    ]
    assert _resolve_book_title("data/books/kant.epub", store) == ""


def test_resolve_book_title_returns_empty_when_book_source_is_none():
    store = MagicMock()
    assert _resolve_book_title(None, store) == ""
    store.list_book_titles.assert_not_called()


def test_resolve_book_title_returns_empty_on_store_exception():
    store = MagicMock()
    store.list_book_titles.side_effect = Exception("chroma down")
    assert _resolve_book_title("data/books/kant.epub", store) == ""


def test_title_from_docs_returns_first_title():
    doc1 = MagicMock()
    doc1.metadata = {"book_title": "纯粹理性批判"}
    doc2 = MagicMock()
    doc2.metadata = {"book_title": "精神现象学"}
    assert _title_from_docs([doc1, doc2]) == "纯粹理性批判"


def test_title_from_docs_skips_docs_without_title():
    doc1 = MagicMock()
    doc1.metadata = {}
    doc2 = MagicMock()
    doc2.metadata = {"book_title": "精神现象学"}
    assert _title_from_docs([doc1, doc2]) == "精神现象学"


def test_title_from_docs_returns_empty_when_no_docs():
    assert _title_from_docs([]) == ""
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/agents/test_orchestrator_state.py -v -k "resolve_book_title or title_from_docs"
```

Expected: `ImportError` — `_resolve_book_title` and `_title_from_docs` don't exist yet.

- [ ] **Step 3: Add helpers to orchestrator**

In `backend/agents/orchestrator_agent.py`, add after the imports section (before `@dataclass class GraphDeps`):

```python
# ---------------------------------------------------------------------------
# 书名解析工具函数
# ---------------------------------------------------------------------------

def _resolve_book_title(book_source: str | None, store) -> str:
    """从 ChromaDB 元数据中查找当前书的人类可读书名。未找到返回 ''。

    Performance note: list_book_titles() fetches all metadata from ChromaDB on each
    call. For a single-user small-library deployment this is acceptable.
    If the library grows beyond ~50 books, cache the title map at graph
    construction time and invalidate on POST /books/upload.

    Edge case: returns '' when book_source is set but the book has no 'book_title'
    metadata. In this case modify_plan will return an early error string.
    """
    if not book_source:
        return ""
    try:
        for entry in store.list_book_titles():
            if entry.get("source") == book_source:
                return entry.get("book_title", "")
    except Exception:
        pass
    return ""


def _title_from_docs(docs) -> str:
    """从检索结果元数据中提取书名（deepread_book 自动笔记 hook 的 fallback）。"""
    for doc in docs:
        title = (doc.metadata or {}).get("book_title", "")
        if title:
            return title
    return ""
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/agents/test_orchestrator_state.py -v -k "resolve_book_title or title_from_docs"
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/agents/orchestrator_agent.py tests/agents/test_orchestrator_state.py
git commit -m "feat: add _resolve_book_title and _title_from_docs helpers"
```

---

### Task 2: Update `_build_supervisor_tools` — remove book params from tool signatures

**Files:**
- Modify: `backend/agents/orchestrator_agent.py`
- Test: `tests/agents/test_orchestrator_state.py`

- [ ] **Step 1: Write failing unit tests for tool closure injection**

First, augment the **existing** import block at the top of `tests/agents/test_orchestrator_state.py` (lines 1–4) — do not add a second import statement:

```python
from backend.agents.orchestrator_agent import (
    GraphState,
    RequestContext,
    _build_supervisor_tools,   # add this line
    _resolve_book_title,       # add this line (already added in Task 1)
    _title_from_docs,          # add this line (already added in Task 1)
)
from unittest.mock import MagicMock   # add this line
```

Then add the following tests to the bottom of `tests/agents/test_orchestrator_state.py`:

```python
def _make_deps():
    """Build a GraphDeps-like mock with all required agents."""
    deps = MagicMock()

    deepread_result = MagicMock()
    deepread_result.citations = []
    deepread_result.retrieved_docs = []
    deepread_result.answer = "deep answer"
    deps.deepread_agent.run.return_value = deepread_result

    plan_result = MagicMock()
    plan_result.citations = []
    plan_result.retrieved_docs = []
    plan_result.answer = "plan answer"
    deps.plan_agent.run.return_value = plan_result

    recommend_result = MagicMock()
    recommend_result.answer = "recommend answer"
    deps.recommend_agent.run.return_value = recommend_result

    return deps


def test_deepread_tool_uses_closure_book_source():
    deps = _make_deps()
    ctx = RequestContext()
    tools = _build_supervisor_tools(
        deps, ctx, memory_context="", thread_id="t1",
        book_source="data/books/kant.epub", book_title="纯粹理性批判"
    )
    deepread = next(t for t in tools if t.name == "deepread_book")
    deepread.invoke({"query": "什么是先验感性论"})
    deps.deepread_agent.run.assert_called_once_with(
        query="什么是先验感性论",
        book_source="data/books/kant.epub",
        memory_context="",
    )


def test_modify_plan_tool_uses_closure_book_title():
    deps = _make_deps()
    ctx = RequestContext()
    tools = _build_supervisor_tools(
        deps, ctx, memory_context="", thread_id="t1",
        book_source="data/books/kant.epub", book_title="纯粹理性批判"
    )
    plan = next(t for t in tools if t.name == "modify_plan")
    plan.invoke({"query": "增加第三章", "action": "extend"})
    deps.plan_agent.run.assert_called_once_with(
        query="增加第三章",
        book_title="纯粹理性批判",
        action="extend",
        memory_context="",
    )


def test_modify_plan_returns_error_when_no_book_title():
    deps = _make_deps()
    ctx = RequestContext()
    tools = _build_supervisor_tools(
        deps, ctx, memory_context="", thread_id="t1",
        book_source=None, book_title=""
    )
    plan = next(t for t in tools if t.name == "modify_plan")
    result = plan.invoke({"query": "增加第三章"})
    assert "未打开" in result
    deps.plan_agent.run.assert_not_called()


def test_recommend_tool_uses_closure_book_title():
    deps = _make_deps()
    ctx = RequestContext()
    tools = _build_supervisor_tools(
        deps, ctx, memory_context="", thread_id="t1",
        book_source="data/books/kant.epub", book_title="纯粹理性批判"
    )
    rec = next(t for t in tools if t.name == "recommend_books")
    rec.invoke({"query": "推荐相似书", "recommend_type": "similar"})
    deps.recommend_agent.run.assert_called_once_with(
        query="推荐相似书",
        current_book="纯粹理性批判",
        memory_context="",
        recommend_type="similar",
    )


def test_recommend_tool_passes_empty_string_when_no_book():
    deps = _make_deps()
    ctx = RequestContext()
    tools = _build_supervisor_tools(
        deps, ctx, memory_context="", thread_id="t1",
        book_source=None, book_title=""
    )
    rec = next(t for t in tools if t.name == "recommend_books")
    rec.invoke({"query": "推荐好书"})
    call_kwargs = deps.recommend_agent.run.call_args.kwargs
    assert call_kwargs["current_book"] == ""
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/agents/test_orchestrator_state.py -v -k "tool_uses_closure or returns_error_when_no"
```

Expected: `TypeError` — `_build_supervisor_tools` doesn't accept `book_source` / `book_title` yet.

- [ ] **Step 3: Rewrite `_build_supervisor_tools`**

Replace the entire `_build_supervisor_tools` function in `backend/agents/orchestrator_agent.py` with:

```python
def _build_supervisor_tools(
    deps: GraphDeps,
    ctx: RequestContext,
    memory_context: str,
    thread_id: str,
    book_source: str | None,
    book_title: str,
):
    """构建 3 个子 Agent 工具，通过闭包绑定请求上下文和当前书籍信息。"""

    @tool
    def deepread_book(query: str) -> str:
        """对当前打开的书籍进行深度精读、解析和问答。
        支持多次检索以验证证据充分性。每次调用后系统会自动将问答记录到笔记。
        """
        result = deps.deepread_agent.run(
            query=query,
            book_source=book_source,
            memory_context=memory_context,
        )
        ctx.citations = result.citations
        ctx.retrieved_docs_count = len(result.retrieved_docs)
        ctx.intent = "deepread"
        print(f"[Supervisor.tool] deepread_book done, hits={len(result.retrieved_docs)}", file=sys.stdout)

        # 自动触发笔记 hook：优先使用闭包中的 book_title，fallback 到文档元数据
        resolved_title = book_title or _title_from_docs(result.retrieved_docs)
        if resolved_title:
            try:
                deps.notes_agent.process_qa(query, result.answer, resolved_title)
            except Exception as e:
                print(f"[Supervisor.tool] note hook failed: {e}", file=sys.stderr)

        return result.answer

    @tool
    def modify_plan(query: str, action: str = "edit") -> str:
        """修改或扩展当前书籍的阅读计划。
        action: 'edit'（修改内容）或 'extend'（增加内容）
        """
        if not book_title:
            return "当前未打开任何书籍，无法修改计划。"
        safe_action: Literal["edit", "extend"] = "extend" if action == "extend" else "edit"
        result = deps.plan_agent.run(
            query=query,
            book_title=book_title,
            action=safe_action,
            memory_context=memory_context,
        )
        ctx.citations = result.citations
        ctx.retrieved_docs_count += len(result.retrieved_docs)  # PlanEditor 始终返回空列表，+= 为一致性
        ctx.intent = "plan"
        print("[Supervisor.tool] modify_plan done", file=sys.stdout)
        return result.answer

    @tool
    def recommend_books(query: str, recommend_type: str = "discover") -> str:
        """从整个出版世界中推荐值得精读的书籍（不限于本地书库）。
        recommend_type: discover(发现新书) / similar(相似书) / next(下一本) / theme(主题推荐)
        会自动标注书籍是否已在本地书库（✅已在库 / 📥可上传）。
        """
        result = deps.recommend_agent.run(
            query=query,
            current_book=book_title or "",
            memory_context=memory_context,
            recommend_type=recommend_type,  # type: ignore[arg-type]
        )
        ctx.intent = "recommend"
        print("[Supervisor.tool] recommend_books done", file=sys.stdout)
        return result.answer

    return [deepread_book, modify_plan, recommend_books]
```

- [ ] **Step 4: Run new tests**

```bash
pytest tests/agents/test_orchestrator_state.py -v -k "tool_uses_closure or returns_error_when_no or passes_empty_string"
```

Expected: 5 tests PASS.

- [ ] **Step 5: Run full test suite to check regressions**

```bash
pytest tests/ -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add backend/agents/orchestrator_agent.py tests/agents/test_orchestrator_state.py
git commit -m "refactor: remove book params from supervisor tool signatures, inject via closure"
```

---

### Task 3: Update `_react_supervisor` call site and `SUPERVISOR_SYSTEM` prompt

**Files:**
- Modify: `backend/agents/orchestrator_agent.py`

No new tests needed — the call-site wiring is covered by the integration path through `_react_supervisor`, which is already exercised by the tool tests added in Task 2 (they call `_build_supervisor_tools` directly). The prompt change is tested implicitly.

- [ ] **Step 1: Update `SUPERVISOR_SYSTEM`**

In `backend/agents/orchestrator_agent.py`, find the `SUPERVISOR_SYSTEM` string and change rule 2 from:

```
2. 修改计划时：直接调用 modify_plan，传入书名和修改要求
```

To:

```
2. 修改计划时：直接调用 modify_plan，只需传入用户的修改要求（书籍已自动关联）
```

- [ ] **Step 2: Update `_react_supervisor` to extract and pass book values**

Find the section inside `_react_supervisor` that currently reads:

```python
ctx = RequestContext()

# 构建 supervisor ReAct agent（每次请求重建以捕获最新上下文）
tools = _build_supervisor_tools(deps, ctx, memory_context, thread_id)
```

Replace with:

```python
ctx = RequestContext()

# 解析当前书籍信息（file path → human-readable title）
book_source = state.get("book_source")
book_title = _resolve_book_title(book_source, store)

# 构建 supervisor ReAct agent（每次请求重建以捕获最新上下文）
tools = _build_supervisor_tools(deps, ctx, memory_context, thread_id, book_source, book_title)
```

Note: `store` is already in scope here via closure from `build_minimal_supervisor_graph`.

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add backend/agents/orchestrator_agent.py
git commit -m "refactor: wire book_source/book_title into react_supervisor, update system prompt"
```

---

### Task 4: Smoke test end-to-end via API

This task is manual — no automated test needed. It verifies the full path from API to tool agents with a real (or mock) ChromaDB.

- [ ] **Step 1: Start the server**

```bash
uvicorn backend.main:app --reload
```

- [ ] **Step 2: Send a chat request with book_source set**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "这本书讲了什么", "book_source": "data/books/kant.epub", "thread_id": "test-1"}'
```

Expected: response JSON with `answer`, `intent: "deepread"`.

- [ ] **Step 3: Send a plan modification request**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "把第三章的进度标记为完成", "book_source": "data/books/kant.epub", "thread_id": "test-1"}'
```

Expected: response JSON with `intent: "plan"`, answer contains updated plan content.

- [ ] **Step 4: Send a request with no book_source**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "推荐几本好书", "thread_id": "test-2"}'
```

Expected: response with `intent: "recommend"`, no errors.

- [ ] **Step 5: Verify logs show correct book values**

In server stdout, confirm lines like:
```
[Supervisor.tool] deepread_book done, hits=5
[NoteAgent] processed Q&A for 《纯粹理性批判》, concepts=[...]
```

The book title in NoteAgent output should be the human-readable title, not a file path.
