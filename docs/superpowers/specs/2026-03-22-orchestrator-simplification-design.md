# Orchestrator Simplification Design

**Date:** 2026-03-22
**Status:** Approved

## Background

The orchestrator (`backend/agents/orchestrator_agent.py`) is a ReAct supervisor that routes user requests to tool agents. The current code is hard to reason about because `book_source` lives in `GraphState` but is never explicitly injected into tool calls — the LLM is expected to infer and pass it correctly, which is unreliable.

The product context clarifies the situation: the user always has **one book open** in the left-side reader. All AI interactions on the right side (chat, plan, notes) operate in the context of that single book. `book_source` is therefore a **session-level constant**, not a decision the LLM should make.

## Goal

Make the orchestrator a thin, unambiguous ReAct layer:

- LLM decides **what to ask** and **which tool to call**
- Session context (`book_source`, `book_title`, `memory_context`) is injected via closure — not inferred by LLM
- Tool signatures only expose parameters the LLM genuinely needs to reason about

## Semantic Clarification: `book_source` vs `book_title`

These are two distinct values that must both be tracked:

| Field | Example | Used by |
|---|---|---|
| `book_source` | `data/books/kant_critique.epub` | ChromaDB `source` filter, `DeepReadAgent`, `get_chapter_structure` |
| `book_title` | `纯粹理性批判` | Plan file naming (`find_by_book`), plan display strings, note hook |

Currently `book_source` in `GraphState` is the ChromaDB file path. `PlanEditor.run(book_title=...)` expects a human-readable display title — a different string. The refactor must not silently pass a file path where a title is expected.

**Resolution:** Add `book_title` as a second session-level value derived from `book_source` at the `_react_supervisor` entry point, using `store.list_book_titles()` to look up the title for the current source. Both values are then closed over into the tools that need them.

```python
def _react_supervisor(state: GraphState, config) -> dict:
    book_source = state.get("book_source")  # file path, e.g. "data/books/kant.epub"
    book_title = _resolve_book_title(book_source, store)  # human-readable title
    ...
    tools = _build_supervisor_tools(deps, ctx, memory_context, thread_id, book_source, book_title)
```

`store` is available here because `_react_supervisor` is already a nested closure inside `build_minimal_supervisor_graph`, which holds `store` in its local scope — the same pattern used for `deps` and `llm`.

```python
def _resolve_book_title(book_source: str | None, store: ChromaStore) -> str:
    """Look up human-readable title from ChromaDB metadata. Returns '' if not found.

    Performance note: list_book_titles() fetches all metadata from ChromaDB on each
    call. For the current single-user small-library deployment this is acceptable.
    If the library grows beyond ~50 books, consider caching the title map at graph
    construction time and invalidating on POST /books/upload.

    Edge case: returns '' when book_source is set but the book has no 'book_title'
    metadata (e.g. ingestion didn't set the field). In this case modify_plan will
    return an early error, which is incorrect but harmless — the book is open but
    its title is unresolvable. This should not occur in normal ingest flows.
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
```

## Architecture

### Graph (unchanged)

```
START → memory_search → react_supervisor → finalize → END
```

- `memory_search`: searches Mem0 for relevant past Q&A, injects `memory_context`
- `react_supervisor`: safety check → resolve book title → build tools → invoke ReAct agent
- `finalize`: saves Q&A to Mem0

### Tool Signatures (simplified)

**Before:**
```python
def deepread_book(query: str, book_source: str = "") -> str
def modify_plan(query: str, book_title: str, action: str = "edit") -> str
def recommend_books(query: str, current_book: str = "", recommend_type: str = "discover") -> str
```

**After:**
```python
def deepread_book(query: str) -> str
def modify_plan(query: str, action: str = "edit") -> str
def recommend_books(query: str, recommend_type: str = "discover") -> str
```

All book-related parameters are removed from tool signatures. They are injected via closure.

### Closure Injection

`_build_supervisor_tools` gains `book_source` and `book_title` parameters:

```python
def _build_supervisor_tools(deps, ctx, memory_context, thread_id, book_source, book_title):

    @tool
    def deepread_book(query: str) -> str:
        """对当前打开的书籍进行深度精读、解析和问答。"""
        result = deps.deepread_agent.run(
            query=query,
            book_source=book_source,      # file path from state
            memory_context=memory_context,
        )
        ctx.citations = result.citations
        ctx.retrieved_docs_count = len(result.retrieved_docs)
        ctx.intent = "deepread"

        # Auto-note hook: prefer closed-over book_title; fall back to doc metadata
        resolved_title = book_title or _title_from_docs(result.retrieved_docs)
        if resolved_title:
            try:
                deps.notes_agent.process_qa(query, result.answer, resolved_title)
            except Exception as e:
                print(f"[Supervisor.tool] note hook failed: {e}", file=sys.stderr)

        return result.answer

    @tool
    def modify_plan(query: str, action: str = "edit") -> str:
        """修改或扩展当前书籍的阅读计划。action: 'edit' 或 'extend'"""
        if not book_title:
            return "当前未打开任何书籍，无法修改计划。"
        safe_action: Literal["edit", "extend"] = "extend" if action == "extend" else "edit"
        result = deps.plan_agent.run(
            query=query,
            book_title=book_title,        # human-readable title from metadata
            action=safe_action,
            memory_context=memory_context,
        )
        ctx.citations = result.citations
        ctx.retrieved_docs_count += len(result.retrieved_docs)
        ctx.intent = "plan"
        return result.answer

    @tool
    def recommend_books(query: str, recommend_type: str = "discover") -> str:
        """从整个出版世界中推荐值得精读的书籍。
        recommend_type: discover / similar / next / theme
        """
        result = deps.recommend_agent.run(
            query=query,
            current_book=book_title or "",  # human-readable, safe to be empty
            memory_context=memory_context,
            recommend_type=recommend_type,
        )
        ctx.intent = "recommend"
        return result.answer

    return [deepread_book, modify_plan, recommend_books]
```

Helper used by the auto-note hook fallback:
```python
def _title_from_docs(docs) -> str:
    for doc in docs:
        title = (doc.metadata or {}).get("book_title", "")
        if title:
            return title
    return ""
```

### None / No-Book-Open Behavior

`book_source` is `None` when the user opens the chat without a book. Each tool handles this gracefully:

| Tool | `book_source=None` behavior |
|---|---|
| `deepread_book` | Passes `book_source=None` to `DeepReadAgent`, which searches across all books |
| `modify_plan` | Returns early with "当前未打开任何书籍，无法修改计划。" |
| `recommend_books` | Passes `current_book=""`, which is already a valid state |

### SUPERVISOR_SYSTEM Prompt Update

The prompt's instruction for `modify_plan` changes from:

> 2. 修改计划时：直接调用 modify_plan，传入书名和修改要求

To:

> 2. 修改计划时：直接调用 modify_plan，只需传入用户的修改要求（书籍已自动关联）

### Reader Context (unchanged)

`selected_text` and `current_chapter` continue to be injected into the user message string:

```python
if selected_text:
    task_content = f"【用户划选的原文片段】：\n{selected_text}\n\n【用户问题】：\n{task_content}"
if current_chapter:
    task_content += f"\n\n【当前阅读章节】：{current_chapter}"
```

### Auto-Note Hook (clarified)

The hook stays in the orchestrator tool closure. After the refactor, `book_title` is available directly from the closure, making the hook more reliable. The doc-metadata fallback is kept for the `book_source=None` case where multiple books may have been searched.

### Cross-Book Note Associations (unaffected)

`NoteAgent._find_associations()` searches the `note_vector_store` across all books using `exclude_book=current_book`. This path is completely independent of `book_source` and is unaffected by this change.

## What Changes

| Location | Change |
|---|---|
| `_build_supervisor_tools` | Add `book_source`, `book_title` params; remove book params from all 3 tool signatures |
| `_react_supervisor` | Extract `book_source` from state; call `_resolve_book_title`; pass both to `_build_supervisor_tools` |
| `_resolve_book_title` | New private helper function |
| `_title_from_docs` | New private helper function |
| `SUPERVISOR_SYSTEM` | Update rule 2 wording as specified above |
| `GraphState` | No change — `book_source` field stays |
| All tool agents | No change |

**Updated call site** (was line 258):
```python
# Before
tools = _build_supervisor_tools(deps, ctx, memory_context, thread_id)

# After
book_source = state.get("book_source")
book_title = _resolve_book_title(book_source, store)
tools = _build_supervisor_tools(deps, ctx, memory_context, thread_id, book_source, book_title)
```

## What Does NOT Change

- Graph topology (`memory_search → react_supervisor → finalize`)
- `RequestContext` side-effect collector pattern
- Message history management (trim to last 8, remove trailing human message)
- `selected_text` / `current_chapter` injection
- Auto-note hook location (stays in orchestrator)
- `invalidate_bm25_caches()` utility
- `run_minimal_graph()` public API

## Known Debt (Out of Scope)

`PlanEditor._build_react_agent()` defines `get_chapter_structure(book_source: str)` as an LLM-visible parameter inside the inner ReAct agent. This is the same class of problem — the inner LLM must infer the file path. This is left for a separate refactor.

## Testing

**Unit test (new pattern):** Instantiate `_build_supervisor_tools` directly with a known `book_source` and `book_title`, call each returned tool with a mocked `deps`, assert that the agent's `run()` was called with the correct book values:

```python
def test_deepread_uses_state_book_source(mock_deps):
    tools = _build_supervisor_tools(mock_deps, RequestContext(), "", "t1", "data/books/kant.epub", "纯粹理性批判")
    deepread = next(t for t in tools if t.name == "deepread_book")
    deepread.invoke({"query": "什么是先验感性论"})
    mock_deps.deepread_agent.run.assert_called_once_with(
        query="什么是先验感性论",
        book_source="data/books/kant.epub",
        memory_context="",
    )
```

**Integration assertions** to add to `tests/agents/test_orchestrator_state.py`:
1. `deepread_book` uses `book_source` from state, not LLM inference
2. `modify_plan` passes `book_title` (resolved human title) to `PlanEditor.run`
3. `modify_plan` returns early error string when `book_source=None`
4. `recommend_books` passes `current_book=""` when `book_source=None`
