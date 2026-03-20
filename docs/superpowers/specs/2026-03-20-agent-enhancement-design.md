# Agent Enhancement Design
**Date:** 2026-03-20
**Scope:** NoteAgent, RecommendationAgent, ReadingPlanAgent — multi-turn state, per-agent message history, cross-agent pipeline, persistent storage

---

## Background

DeepReadAgent is the most complete agent in the system. The other three agents (NoteAgent, RecommendationAgent, ReadingPlanAgent) lack multi-turn awareness, use weaker retrieval, and lose all state on process restart. This document defines the enhancements needed to bring them to parity and add new capabilities.

### Current Gaps

| Agent | Gap |
|---|---|
| NoteAgent | One-shot output; pure similarity_search; no editing; no storage |
| RecommendationAgent | Chunk-level similarity only; no global book catalog; no repeat-avoidance |
| ReadingPlanAgent | No real chapter structure; time estimates have no basis; no plan editing |
| All agents | MemorySaver loses all state on restart; no per-agent message history |

---

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Frontend routing | Output-format-based only; Supervisor does all routing | No explicit mode tabs in UI |
| Multi-turn state | Per-agent message lists in GraphState | Leverages existing LangGraph checkpointer |
| Persistence | SqliteSaver replacing MemorySaver | Zero new dependencies; one-line change |
| Agent memory | GraphState fields, not Mem0 namespaces | Mem0 is for user preference extraction only |
| Cross-agent handoff | `handoff_docs` serialized as `list[dict]` in GraphState | JSON-safe for SqliteSaver; reconstructed to Document at consumer |
| Note/Plan storage | Pluggable Storage interface (local → S3) | Avoids storing large text in GraphState |

---

## Section 1: Architecture

### Flow

```
User Input
  ↓
memory_search node  →  Mem0.search() → memory_context (user preferences)
  ↓
Supervisor node
  - Safety check
  - Intent classification (intent + action + source_agent)
  - Validate handoff freshness before populating handoff_docs
  - Write HumanMessage(task) to target agent's messages list as delta
  ↓
[deepread | notes | plan | recommend] node
  - Read own {agent}_messages as conversation context
  - Use handoff_docs if present (skip retrieval)
  - Generate output
  - Write AIMessage(output) to own {agent}_messages
  - Write {agent}_last_docs (serialized) and {agent}_last_turn_index for downstream
  ↓
Supervisor node  →  detects answer present → finalize
  ↓
finalize node  (closure inside build_minimal_supervisor_graph, captures deps)
  - Write AIMessage(final_answer) to global messages
  - Mem0.add_qa() to extract user preferences
```

### Message List Separation

```
messages             ← User ↔ Supervisor global conversation
deepread_messages    ← Supervisor task ↔ DeepReadAgent output
notes_messages       ← Supervisor task ↔ NoteAgent output
plan_messages        ← Supervisor task ↔ ReadingPlanAgent output
recommend_messages   ← Supervisor task ↔ RecommendationAgent output
```

All message lists use LangGraph's `add_messages` reducer and are persisted by SqliteSaver under the same `thread_id`. **All node functions return delta dicts, not the full mutated state.**

---

## Section 2: GraphState Changes

### Serialization constraint

`SqliteSaver` serializes GraphState to JSON. All fields must contain JSON-safe values. `langchain_core.documents.Document` objects are NOT stored directly in state. Agent nodes serialize retrieved docs to `list[dict]` (via `doc.model_dump()`) before writing to state, and deserialize back to `Document` objects when reading. `Document.metadata` must contain only primitive types (str, int, float, list of primitives) — this is enforced at ingest time in `ChromaStore`.

### Typed metadata pointers

```python
class NoteOutputMeta(TypedDict):
    note_id: str
    book_title: str
    topics: list[str]
    storage_path: str
    created_at: str   # ISO 8601

class PlanOutputMeta(TypedDict):
    plan_id: str
    book_titles: list[str]
    plan_type: str
    storage_path: str
    created_at: str   # ISO 8601
    progress_summary: str
```

### Full GraphState

```python
class GraphState(TypedDict, total=False):
    # ── Existing fields (unchanged) ─────────────────────────────────
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str
    book_source: str | None
    intent: Literal["recommend", "deepread", "notes", "plan"] | None
    intent_reason: str
    safety_ok: bool
    safety_reason: str
    memory_context: str
    answer: str
    citations: list[Any]
    retrieved_docs_count: int
    next: Literal["deepread", "notes", "plan", "recommend", "finalize", "end"]

    # Agent task fields (existing, unchanged)
    deepread_query: str
    deepread_book_source: str | None
    notes_query: str
    notes_book_source: str | None
    notes_raw_text: str | None
    plan_query: str
    plan_book_source: str | None
    recommend_query: str

    # ── New: per-agent message lists ────────────────────────────────
    deepread_messages:   Annotated[list[AnyMessage], add_messages]
    notes_messages:      Annotated[list[AnyMessage], add_messages]
    plan_messages:       Annotated[list[AnyMessage], add_messages]
    recommend_messages:  Annotated[list[AnyMessage], add_messages]

    # ── New: cross-agent pipeline ────────────────────────────────────
    # Stored as list[dict] (JSON-safe), NOT list[Document]
    handoff_docs:        list[dict] | None
    handoff_source:      str | None   # "deepread" / "notes" / etc.
    handoff_turn_index:  int | None   # turn index when handoff was produced

    # Last retrieved docs per agent — serialized as list[dict] for JSON safety
    deepread_last_docs:   list[dict] | None
    notes_last_docs:      list[dict] | None
    plan_last_docs:       list[dict] | None
    recommend_last_docs:  list[dict] | None

    # Turn index incremented by Supervisor each time it dispatches a new task
    current_turn_index: int

    # ── New: action type ─────────────────────────────────────────────
    action: Literal["new", "edit", "extend"] | None
    # new    → fresh task, no dependency on history
    # edit   → modify a specific part of the last output
    # extend → append to the last output

    # ── New: structured output metadata pointers ─────────────────────
    notes_last_output:  NoteOutputMeta | None
    plan_last_output:   PlanOutputMeta | None

    # ── New: per-agent turn indices (for handoff staleness guard) ───
    deepread_last_turn_index:   int | None
    notes_last_turn_index:      int | None
    plan_last_turn_index:       int | None
    recommend_last_turn_index:  int | None

    # ── New: progress tracking ───────────────────────────────────────
    plan_progress: Annotated[list[str], lambda a, b: a + b]
    # Accumulates explicitly completed section identifiers
    # e.g. ["先验感性论", "先验分析论·概念分析论"]
    # Uses list-append reducer so completed sections accumulate across turns
    # Supervisor appends to this when it detects a progress-update intent
```

---

## Section 3: Supervisor Upgrade

### IntentSchema Extension

```python
class IntentSchema(BaseModel):
    intent: Literal["recommend", "deepread", "notes", "plan"]
    action: Literal["new", "edit", "extend"] = "new"
    reason: str
    book_source: str | None = None
    source_agent: Literal["deepread", "notes", "plan", "recommend"] | None = None
    # Populated when user implies cross-agent pipeline AND most recent turn
    # in that agent's messages list is the immediately preceding turn.
    # e.g. "把这次精读做成笔记" → intent=notes, source_agent="deepread"
    notes_format: Literal["structured", "summary", "qa", "timeline"] | None = None
    recommend_type: Literal["discover", "similar", "next", "theme"] | None = None
    plan_type: Literal["single_deep", "multi_theme", "research"] | None = None
    is_progress_update: bool = False
    # True when user reports reading progress ("XX我读完了")
```

### classify_intent Upgraded Signature

```python
def classify_intent(
    user_input: str,
    agent_last_turns: dict[str, str],
    # Keys: "deepread", "notes", "plan", "recommend"
    # Values: last AIMessage content from that agent (empty string if none)
    # Passed to LLM as context for detecting cross-agent references and edit intent
) -> IntentSchema: ...

def _extract_agent_last_turns(state: GraphState) -> dict[str, str]:
    """Extract the last AIMessage content from each agent's messages list."""
    result = {}
    for agent in ("deepread", "notes", "plan", "recommend"):
        msgs = state.get(f"{agent}_messages") or []
        # Scan in reverse to find the most recent AIMessage
        last_ai = ""
        for m in reversed(msgs):
            if getattr(m, "type", None) == "ai":
                last_ai = (getattr(m, "content", "") or "")[:300]
                break
        result[agent] = last_ai
    return result
```

**Prompt additions** (appended to existing prompt):
```
可参考的子 Agent 最近输出摘要（判断用户是否在引用上一轮结果）：
- deepread 最近回复：{agent_last_turns["deepread"][:300]}
- notes 最近回复：{agent_last_turns["notes"][:300]}
- plan 最近回复：{agent_last_turns["plan"][:300]}
- recommend 最近回复：{agent_last_turns["recommend"][:300]}

额外判断规则：
- 如果用户说"把这次精读/笔记/推荐...做成..."，source_agent 填对应 agent
- 如果用户说"修改/更新/调整/把...改成..."，action=edit
- 如果用户说"再加/继续/补充..."，action=extend
- 如果用户说"XX章节我读完了/已读"，is_progress_update=true，intent=plan
```

### handoff_docs Staleness Guard

Before populating `handoff_docs`, the supervisor validates that the source agent's last output is from the immediately preceding turn:

```python
# In supervisor_node — returns delta dict, NOT mutated state

def supervisor_node(state: GraphState) -> dict:
    # ... safety check, intent classification ...

    patch = {
        "intent": result.intent,
        "action": result.action,
        "current_turn_index": (state.get("current_turn_index") or 0) + 1,
    }

    # Staleness guard for cross-agent handoff
    if result.source_agent:
        source_last_turn = state.get(f"{result.source_agent}_last_turn_index")
        current_turn = state.get("current_turn_index") or 0
        if source_last_turn is not None and current_turn - source_last_turn <= 1:
            # Immediately preceding turn — handoff is fresh
            patch["handoff_docs"] = state.get(f"{result.source_agent}_last_docs")
            patch["handoff_source"] = result.source_agent
            patch["handoff_turn_index"] = source_last_turn
        else:
            # Stale or missing — do not populate handoff, agent will do its own retrieval
            patch["handoff_docs"] = None
            patch["handoff_source"] = None

    # Progress update path
    if result.is_progress_update:
        # Extract completed section from user_input; append to plan_progress
        # (exact extraction delegated to ReadingPlanAgent)
        patch["plan_query"] = user_input
        patch["plan_messages"] = [HumanMessage(content=user_input)]
        patch["next"] = "plan"
        return patch

    # Normal dispatch — write HumanMessage as delta to target agent's messages
    task_msg = HumanMessage(content=build_task_message(user_input, result))
    if result.intent == "notes":
        patch["notes_messages"] = [task_msg]
        patch["notes_query"] = user_input
        patch["notes_book_source"] = result.book_source or state.get("book_source")
        patch["next"] = "notes"
    elif result.intent == "plan":
        patch["plan_messages"] = [task_msg]
        patch["plan_query"] = user_input
        patch["plan_book_source"] = result.book_source or state.get("book_source")
        patch["next"] = "plan"
    elif result.intent == "recommend":
        patch["recommend_messages"] = [task_msg]
        patch["recommend_query"] = user_input
        patch["next"] = "recommend"
    else:
        patch["deepread_messages"] = [task_msg]
        patch["deepread_query"] = user_input
        patch["deepread_book_source"] = result.book_source or state.get("book_source")
        patch["next"] = "deepread"

    return patch
```

**Key rule:** All supervisor and agent node functions return **delta dicts** (the patch), not the full mutated state. The `add_messages` reducer on `Annotated[list[AnyMessage], add_messages]` fields accumulates correctly when the delta contains a list value.

### finalize Node

`_finalize` is a **closure** defined inside `build_minimal_supervisor_graph`, capturing `deps` from the enclosing scope. It is not a module-level function.

```python
# Inside build_minimal_supervisor_graph():
def _finalize(state: GraphState) -> dict:
    patch = {"next": "end"}
    if state.get("answer"):
        patch["messages"] = [AIMessage(content=state["answer"])]
    if deps.mem0 and state.get("answer") and state.get("user_input"):
        deps.mem0.add_qa(state["user_input"], state["answer"])
    return patch
```

---

## Section 4: Agent Enhancements

### 4.1 NoteAgent

**Constructor changes:**
- `note_storage: NoteStorage` — injected via GraphDeps
- `HybridRetriever` constructed in `__init__` (not per `run()` call) to avoid BM25 index rebuild cost

**`run()` signature:**
```python
def run(
    self,
    *,
    query: str,
    book_source: str | None = None,
    raw_text: str | None = None,
    memory_context: str = "",
    notes_messages: list[AnyMessage] | None = None,
    action: Literal["new", "edit", "extend"] = "new",
    notes_format: Literal["structured", "summary", "qa", "timeline"] = "structured",
    handoff_docs: list[dict] | None = None,
    storage_path: str | None = None,   # required when action=edit or action=extend
) -> NoteResult: ...
```

`notes_node()` reads `storage_path` from state before calling `run()`. It accepts `deps` and `thread_id` passed in by the `_notes` closure (see Section 3 finalize pattern):
```python
def notes_node(state, *, agent, deps, thread_id) -> dict:
    last = state.get("notes_last_output") or {}
    return agent.run(
        ...,
        action=state.get("action") or "new",
        storage_path=last.get("storage_path"),   # None for action=new
        handoff_docs=state.get("handoff_docs"),
        notes_messages=state.get("notes_messages") or [],
    )
```

**Multi-turn editing logic:**
```
action=new    → retrieve via HybridRetriever (or use handoff_docs) → generate fresh note
action=extend → load full note from NoteStorage via storage_path → append new section
action=edit   → load full note from NoteStorage via storage_path → modify specified section
```

`storage_path` is `None` when `action=new`. If `action=edit/extend` and `storage_path` is `None` (no prior note in this session), fall back to `action=new`.

**Note templates:**

| Template | Use case | Output structure |
|---|---|---|
| `structured` (default) | Systematic learning | Hierarchical headings + bullets + bold concepts |
| `summary` | Quick review | Core thesis / Evidence / Insight (3 sections) |
| `qa` | Exam prep | Q&A card format |
| `timeline` | History / biography | Chronological nodes |

**Retrieval:** HybridRetriever (BM25 + vector), same config as DeepReadAgent. Constructed once in `__init__`.

**Cross-book synthesis:** When `book_source=None` and `handoff_docs=None`, retrieve across all books and generate comparative notes.

**NoteStorage integration (node-level):**

`_notes` is a **closure** inside `build_minimal_supervisor_graph` that wraps `notes_node`, giving it access to `deps` and `thread_id` (from `RunnableConfig`). Pattern:

```python
# Inside build_minimal_supervisor_graph():
def _notes(state: GraphState, config: RunnableConfig) -> dict:
    thread_id = config["configurable"].get("thread_id", "default")
    patch = notes_node(state, agent=deps.notes_agent, deps=deps, thread_id=thread_id)
    return patch

# notes_node (module-level, testable):
def notes_node(state, *, agent, deps, thread_id) -> dict:
    ...
    content = result.answer
    note_id = f"note_{thread_id}_{int(datetime.utcnow().timestamp())}"
    storage_path = deps.note_storage.save(content, note_id)
    return {
        "answer": content,
        "citations": result.citations,
        "retrieved_docs_count": len(result.retrieved_docs),
        "notes_last_output": NoteOutputMeta(
            note_id=note_id,
            book_title=...,
            topics=...,
            storage_path=storage_path,
            created_at=datetime.utcnow().isoformat(),
        ),
        "notes_last_docs": [d.model_dump() for d in result.retrieved_docs],
        "notes_last_turn_index": state.get("current_turn_index") or 0,
        "notes_messages": [AIMessage(content=content)],
    }
```

The same closure pattern applies to `_plan` / `plan_node`.

---

### 4.2 RecommendationAgent

**Constructor changes:**
- `HybridRetriever` constructed in `__init__`
- `_catalog_cache: dict | None = None` — in-memory cache for book catalog summary

**`run()` signature:**
```python
def run(
    self,
    *,
    query: str,
    memory_context: str = "",
    recommend_messages: list[AnyMessage] | None = None,
    recommend_type: Literal["discover", "similar", "next", "theme"] = "discover",
    handoff_docs: list[dict] | None = None,
) -> RecommendationResult: ...
```

**Multi-turn awareness:**
Parse `recommend_messages` to extract previously recommended titles → pass as exclusion list to LLM. Detect feedback patterns ("太难了" / "再来几本") and adjust `recommend_type` internally.

**Global book catalog view with caching:**
```python
def _get_catalog_summary(self) -> str:
    sources = self.store.list_sources()
    cache_key = hash(tuple(sorted(sources)))
    if self._catalog_cache and self._catalog_cache.get("key") == cache_key:
        return self._catalog_cache["summary"]
    summary = self._build_catalog_summary(sources[:30])
    self._catalog_cache = {"key": cache_key, "summary": summary}
    return summary

def _build_catalog_summary(self, sources: list[str]) -> str:
    # Single similarity_search per book with a generic query ("introduction overview")
    # and source filter, limit=2 — results in at most 30 queries total at agent init.
    # Each book contributes a one-line entry: "《title》 / author — snippet[:100]"
    lines = []
    for source in sources:
        docs = self.store.similarity_search("introduction overview", k=2,
                                            filter={"source": source})
        if docs:
            meta = docs[0].metadata or {}
            title = meta.get("book_title") or source
            author = meta.get("author") or "未知作者"
            snippet = (docs[0].page_content or "")[:100].replace("\n", " ")
            lines.append(f"《{title}》/ {author} — {snippet}")
    return "\n".join(lines)
```

**Mem0 preference injection:** Parse `memory_context` into structured profile section in system prompt.

**Recommendation sub-types:**

| Type | Example trigger | Strategy |
|---|---|---|
| `discover` | "推荐几本小众好书" | Broad catalog scan |
| `similar` | "像这本一样的" | Current book as anchor for similarity |
| `next` | "读完这本读什么" | Difficulty progression or thematic extension |
| `theme` | "关于时间哲学的书" | Topic-focused retrieval across full catalog |

**Node writes back to state:**
```python
return {
    "answer": result.answer,
    "citations": result.citations,
    "retrieved_docs_count": len(result.retrieved_docs),
    "recommend_last_docs": [d.model_dump() for d in result.retrieved_docs],
    "recommend_last_turn_index": state.get("current_turn_index") or 0,
    "recommend_messages": [AIMessage(content=result.answer)],
}
```

---

### 4.3 ReadingPlanAgent

**Constructor changes:**
- `plan_storage: PlanStorage` — injected via GraphDeps
- `HybridRetriever` constructed in `__init__`

**`run()` signature:**
```python
def run(
    self,
    *,
    query: str,
    book_source: str | None = None,
    memory_context: str = "",
    plan_messages: list[AnyMessage] | None = None,
    action: Literal["new", "edit", "extend"] = "new",
    plan_type: Literal["single_deep", "multi_theme", "research"] = "single_deep",
    handoff_docs: list[dict] | None = None,
    storage_path: str | None = None,   # required when action=edit or action=extend
    plan_progress: list[str] | None = None,  # completed section identifiers
) -> ReadingPlanResult: ...
```

`plan_node()` reads `storage_path` and `plan_progress` from state before calling `run()`. It accepts `deps` and `thread_id` passed in by the `_plan` closure (same pattern as `notes_node`):
```python
def plan_node(state, *, agent, deps, thread_id) -> dict:
    last = state.get("plan_last_output") or {}
    return agent.run(
        ...,
        action=state.get("action") or "new",
        storage_path=last.get("storage_path"),
        plan_progress=state.get("plan_progress") or [],
        handoff_docs=state.get("handoff_docs"),
        plan_messages=state.get("plan_messages") or [],
    )
```

**Multi-turn plan editing:**
```
action=new    → generate fresh plan → save via PlanStorage
action=edit   → load plan from PlanStorage via storage_path → modify → save new version
action=extend → load plan → add books or extend timeline → save new version
```

**Real chapter structure extraction:**
```python
def _extract_chapter_structure(self, book_source: str) -> list[ChapterInfo]:
    # Fetch all chunks for this book, deduplicate by chapter_title
    # Sum page_content lengths per chapter for word count estimate
    # Return sorted list of ChapterInfo(title, estimated_chars)
```

Reading time formula: Chinese ~300 chars/min, English ~200 words/min.

**Plan sub-types:**

| Type | Use case | Output structure |
|---|---|---|
| `single_deep` | Deep-read one book | Chapter-by-chapter schedule + reading goals |
| `multi_theme` | Thematic reading list | Cross-book interleaved reading route |
| `research` | Research a topic | Key chapters across books + annotation targets |

**Progress tracking:**
`plan_progress` is a `list[str]` of completed section identifiers passed in from GraphState. When `action=new/edit/extend`, the agent filters out completed sections from the schedule. When the supervisor detects `is_progress_update=True`, it routes to the plan agent which extracts the completed section name from user input and appends it to `plan_progress` in the returned delta.

**Node writes back to state:**
```python
content = result.answer
plan_id = f"plan_{thread_id}_{timestamp}"
storage_path = deps.plan_storage.save(content, plan_id)
return {
    "answer": content,
    "citations": result.citations,
    "retrieved_docs_count": len(result.retrieved_docs),
    "plan_last_output": PlanOutputMeta(
        plan_id=plan_id,
        book_titles=...,
        plan_type=plan_type,
        storage_path=storage_path,
        created_at=datetime.utcnow().isoformat(),
        progress_summary=f"{len(plan_progress or [])} sections completed",
    ),
    "plan_last_docs": [d.model_dump() for d in result.retrieved_docs],
    "plan_last_turn_index": state.get("current_turn_index") or 0,
    "plan_messages": [AIMessage(content=content)],
    # Append newly completed section if this was a progress update:
    "plan_progress": newly_completed_sections or [],  # add_messages-style append via reducer
}
```

> Note: `plan_progress` needs a list-append reducer in GraphState (`Annotated[list[str], lambda a, b: a + b]`) so sections accumulate across turns without overwriting.

---

## Section 5: Cross-Agent Pipeline

### Trigger examples

| User says | Supervisor detects | Pipeline |
|---|---|---|
| "把这次精读做成笔记" | intent=notes, source_agent="deepread" | deepread_last_docs → NoteAgent |
| "根据刚才的推荐制定计划" | intent=plan, source_agent="recommend" | recommend_last_docs → ReadingPlanAgent |

### Staleness rule

`source_agent` is only trusted when `{source_agent}_last_turn_index == current_turn_index - 1` (immediately preceding turn). Otherwise `handoff_docs` is set to `None` and the target agent performs its own retrieval.

### Data flow

```
Agent node finishes:
  patch["deepread_last_docs"] = [d.model_dump() for d in docs]   # JSON-safe
  patch["deepread_last_turn_index"] = current_turn_index

Supervisor detects cross-agent (fresh):
  patch["handoff_docs"] = state["deepread_last_docs"]       # already list[dict]
  patch["handoff_source"] = "deepread"

Target agent run():
  if handoff_docs:
      docs = [Document(**d) for d in handoff_docs]  # deserialize
  else:
      docs = self.retriever.search(query, filter=filter_)
```

---

## Section 6: Storage Layer

### NoteStorage

```python
# backend/storage/note_storage.py

class NoteStorage(Protocol):
    def save(self, content: str, note_id: str) -> str: ...   # returns storage_path
    def load(self, storage_path: str) -> str: ...
    def list(self, prefix: str = "") -> list[str]: ...       # list storage_paths
    def delete(self, storage_path: str) -> None: ...

class LocalNoteStorage:
    """Stores notes as .md files under data/notes/"""
    def __init__(self, root: Path): ...

class S3NoteStorage:
    """Stores notes in S3-compatible object storage"""
    def __init__(self, bucket: str, prefix: str = "notes/"): ...
```

### PlanStorage

```python
# backend/storage/plan_storage.py

class PlanStorage(Protocol):
    def save(self, content: str, plan_id: str) -> str: ...
    def load(self, storage_path: str) -> str: ...
    def list(self, prefix: str = "") -> list[str]: ...
    def delete(self, storage_path: str) -> None: ...

class LocalPlanStorage:
    """Stores plans as .md files under data/plans/"""
    def __init__(self, root: Path): ...

class S3PlanStorage:
    """Stores plans in S3-compatible object storage"""
    def __init__(self, bucket: str, prefix: str = "plans/"): ...
```

### GraphDeps Update

```python
@dataclass
class GraphDeps:
    deepread_agent:   DeepReadAgent
    notes_agent:      NoteAgent
    plan_agent:       ReadingPlanAgent
    recommend_agent:  RecommendationAgent
    mem0:             Mem0Store | None = None
    note_storage:     NoteStorage = field(default_factory=lambda: LocalNoteStorage(Path("data/notes")))
    plan_storage:     PlanStorage = field(default_factory=lambda: LocalPlanStorage(Path("data/plans")))
```

### Persistence: SqliteSaver

```python
# orchestrator_agent.py

# Before:
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# After:
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("data/checkpoints.db")
```

**Dependency note:** `langgraph-checkpoint-sqlite` must be added to `requirements.txt` as a separate PyPI package. Verify the exact API (`from_conn_string` vs constructor) against the pinned LangGraph version in `requirements.txt` before implementation.

All GraphState fields (per-agent message lists, metadata pointers, progress lists) are automatically persisted under `thread_id`. Process restarts do not lose conversation history.

---

## Config Changes

```env
# .env additions
NOTE_STORAGE_DIR=data/notes
PLAN_STORAGE_DIR=data/plans
```

---

## Full File Change List

| File | Change |
|---|---|
| `backend/agents/orchestrator_agent.py` | GraphState: ~15 new fields incl. typed metadata, turn indices, plan_progress; IntentSchema: 4 new fields; classify_intent: new signature + prompt; Supervisor: delta-return pattern, staleness guard, progress update path; finalize: delta-return, writes to messages; SqliteSaver; GraphDeps adds storage |
| `backend/agents/note_agent.py` | Accept notes_messages + action + notes_format + handoff_docs + storage_path; HybridRetriever in __init__; action mode branching; NoteStorage integration in notes_node |
| `backend/agents/recommendation_agent.py` | Accept recommend_messages + recommend_type + handoff_docs; HybridRetriever in __init__; catalog cache; repeat-avoidance; structured Mem0 preference injection |
| `backend/agents/reading_plan_agent.py` | Accept plan_messages + action + plan_type + handoff_docs + storage_path + plan_progress; HybridRetriever in __init__; chapter structure extraction; word-count time estimation; action mode branching; PlanStorage integration in plan_node |
| `backend/agents/deepread_agent.py` | deepread_node writes deepread_last_docs + deepread_last_turn_index to returned delta |
| `backend/storage/note_storage.py` | New: NoteStorage Protocol + LocalNoteStorage + S3NoteStorage stub |
| `backend/storage/plan_storage.py` | New: PlanStorage Protocol + LocalPlanStorage + S3PlanStorage stub |
| `backend/storage/__init__.py` | New |
| `backend/config.py` | Add NOTE_STORAGE_DIR, PLAN_STORAGE_DIR |
| `requirements.txt` | Add langgraph-checkpoint-sqlite |
