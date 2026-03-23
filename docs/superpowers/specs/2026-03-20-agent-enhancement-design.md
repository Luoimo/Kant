# Agent Enhancement Design
**Date:** 2026-03-20
**Scope:** NoteAgent, RecommendationAgent, ReadingPlanAgent — multi-turn state, per-agent message history, multi-agent pipeline, persistent storage

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
| Cross-agent context | Inject previous agent's output text into next agent's task message | Avoids handoff_docs complexity; no raw doc passing |
| Multi-agent pipeline | `pending_agents` queue + `compound_context` in GraphState | Clean sequential execution; easy to reason about |
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
  - Intent classification (intent + action + compound_intents)
  - If compound: fill pending_agents queue
  - Build task message with compound_context injected
  - Write HumanMessage(task) to target agent's messages list
  ↓
[deepread | notes | plan | recommend] node
  - Read own {agent}_messages as full conversation context
  - Generate output
  - Write AIMessage(output) to own {agent}_messages
  ↓
Supervisor node
  - Update compound_context with agent's output
  - Pop next agent from pending_agents if any → route there
  - Otherwise → finalize
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

`SqliteSaver` serializes GraphState to JSON. All fields must contain JSON-safe values. `Document.metadata` must contain only primitive types (str, int, float, list of primitives) — enforced at ingest time in `ChromaStore`. Agents do not store raw `Document` objects in state.

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

    # ── New: multi-agent pipeline ────────────────────────────────────
    pending_agents:    list[str]
    # Queue of agents yet to run in a compound request.
    # e.g. ["recommend", "plan"] for "推荐一本书并生成读书计划"
    # Supervisor pops the first item on each dispatch.

    compound_context:  str | None
    # Accumulated text output from completed agents in the current pipeline.
    # Injected into the next agent's task message for context.

    # ── New: action type ─────────────────────────────────────────────
    action: Literal["new", "edit", "extend"] | None
    # new    → fresh task, no dependency on history
    # edit   → modify a specific part of the last output
    # extend → append to the last output

    # ── New: structured output metadata pointers ─────────────────────
    notes_last_output:  NoteOutputMeta | None
    plan_last_output:   PlanOutputMeta | None

    # ── New: progress tracking ───────────────────────────────────────
    plan_progress: Annotated[list[str], lambda a, b: a + b]
    # Accumulates explicitly completed section identifiers.
    # e.g. ["先验感性论", "先验分析论·概念分析论"]
    # Uses list-append reducer so sections accumulate across turns without overwriting.
    # Supervisor appends when it detects a progress-update intent.
```

**New fields summary:** 8 total (`*_messages` ×4, `pending_agents`, `compound_context`, `action`, `notes_last_output`, `plan_last_output`, `plan_progress`). No turn indices, no last_docs, no handoff fields.

---

## Section 3: Supervisor Upgrade

### IntentSchema Extension

```python
class IntentSchema(BaseModel):
    intent: Literal["recommend", "deepread", "notes", "plan"]
    action: Literal["new", "edit", "extend"] = "new"
    reason: str
    book_source: str | None = None
    compound_intents: list[str] = []
    # For compound single-message requests, lists all agents to run in order.
    # e.g. "推荐一本康德的书并生成读书计划" → ["recommend", "plan"]
    # e.g. "精读这一章并整理成笔记"         → ["deepread", "notes"]
    # Empty for single-agent requests.
    notes_format: Literal["structured", "summary", "qa", "timeline"] | None = None
    recommend_type: Literal["discover", "similar", "next", "theme"] | None = None
    plan_type: Literal["single_deep", "multi_theme", "research"] | None = None
    is_progress_update: bool = False
    # True when user reports reading progress ("XX我读完了")
```

### classify_intent Signature

```python
def classify_intent(
    user_input: str,
    agent_last_turns: dict[str, str],
    # Keys: "deepread", "notes", "plan", "recommend"
    # Values: last AIMessage content from that agent (truncated to 300 chars; "" if none)
) -> IntentSchema: ...

def _extract_agent_last_turns(state: GraphState) -> dict[str, str]:
    """Extract the last AIMessage content from each agent's messages list."""
    result = {}
    for agent in ("deepread", "notes", "plan", "recommend"):
        msgs = state.get(f"{agent}_messages") or []
        last_ai = ""
        for m in reversed(msgs):
            if getattr(m, "type", None) == "ai":
                last_ai = (getattr(m, "content", "") or "")[:300]
                break
        result[agent] = last_ai
    return result
```

**Prompt additions** (appended to existing classify_intent prompt):
```
可参考的子 Agent 最近输出摘要（判断用户是否在引用上一轮结果）：
- deepread 最近回复：{agent_last_turns["deepread"]}
- notes 最近回复：{agent_last_turns["notes"]}
- plan 最近回复：{agent_last_turns["plan"]}
- recommend 最近回复：{agent_last_turns["recommend"]}

额外判断规则：
- 如果用户一句话要求多件事（如"推荐...并制定计划"），compound_intents 填完整链路，如 ["recommend","plan"]
- 如果用户说"修改/更新/调整/把...改成..."，action=edit
- 如果用户说"再加/继续/补充..."，action=extend
- 如果用户说"XX章节我读完了/已读"，is_progress_update=true，intent=plan
- 不支持动态扇出（如"每本书都做笔记"），此类请求 compound_intents 留空，在回复中说明需要分步操作
```

### Supervisor Routing Logic

```python
def supervisor_node(state: GraphState) -> dict:
    # 1. Safety check (unchanged)
    # 2. Intent classification
    agent_last_turns = _extract_agent_last_turns(state)
    result = classify_intent(user_input, agent_last_turns)

    patch = {"intent": result.intent, "action": result.action}

    # 3. answer already present → finalize
    if state.get("answer"):
        patch["next"] = "finalize"
        return patch

    # 4. Compound pipeline init: fill pending_agents on first dispatch
    pending = list(state.get("pending_agents") or [])
    if result.compound_intents and not pending:
        pending = list(result.compound_intents)
        patch["pending_agents"] = pending
        patch["compound_context"] = None

    # 5. Determine which agent to dispatch next
    if pending:
        target = pending.pop(0)
        patch["pending_agents"] = pending   # updated queue (one item consumed)
    else:
        target = result.intent

    # 6. Build task message — inject compound_context from previous agent if present
    task_content = user_input
    ctx = state.get("compound_context")
    if ctx:
        task_content += f"\n\n【前序步骤结果，供参考】：\n{ctx}"
    task_msg = HumanMessage(content=task_content)

    # 7. Dispatch to target agent
    if target == "notes":
        patch["notes_messages"] = [task_msg]
        patch["notes_query"] = user_input
        patch["notes_book_source"] = result.book_source or state.get("book_source")
        patch["next"] = "notes"
    elif target == "plan":
        patch["plan_messages"] = [task_msg]
        patch["plan_query"] = user_input
        patch["plan_book_source"] = result.book_source or state.get("book_source")
        patch["next"] = "plan"
    elif target == "recommend":
        patch["recommend_messages"] = [task_msg]
        patch["recommend_query"] = user_input
        patch["next"] = "recommend"
    else:
        patch["deepread_messages"] = [task_msg]
        patch["deepread_query"] = user_input
        patch["deepread_book_source"] = result.book_source or state.get("book_source")
        patch["next"] = "deepread"

    # 8. Progress update path
    if result.is_progress_update:
        patch["plan_messages"] = [task_msg]
        patch["plan_query"] = user_input
        patch["next"] = "plan"

    return patch
```

### After Each Agent: Update compound_context

Each agent node appends its output summary to `compound_context`. The Supervisor reads this on the next iteration to inject context into the next agent's task:

```python
# Pattern in every agent node's return delta:
existing_ctx = state.get("compound_context") or ""
agent_label  = "推荐结果"   # or "精读结果", "笔记结果", "计划结果"
new_ctx = (existing_ctx + f"\n\n[{agent_label}]\n{content[:500]}").strip()
patch["compound_context"] = new_ctx
```

### finalize Node

`_finalize` is a **closure** defined inside `build_minimal_supervisor_graph`, capturing `deps`.

```python
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
    storage_path: str | None = None,   # required when action=edit or action=extend
) -> NoteResult: ...
```

`_notes` closure passes `deps` and `thread_id`; `notes_node` is module-level and testable:

```python
# Inside build_minimal_supervisor_graph():
def _notes(state: GraphState, config: RunnableConfig) -> dict:
    thread_id = config["configurable"].get("thread_id", "default")
    return notes_node(state, agent=deps.notes_agent, deps=deps, thread_id=thread_id)

# Module-level:
def notes_node(state, *, agent, deps, thread_id) -> dict:
    last = state.get("notes_last_output") or {}
    result = agent.run(
        query=state.get("notes_query") or state.get("user_input") or "",
        book_source=state.get("notes_book_source") or state.get("book_source"),
        raw_text=state.get("notes_raw_text"),
        memory_context=state.get("memory_context") or "",
        notes_messages=state.get("notes_messages") or [],
        action=state.get("action") or "new",
        storage_path=last.get("storage_path"),
        notes_format=...,   # from IntentSchema, passed via state or default
    )
    content = result.answer
    note_id = f"note_{thread_id}_{int(datetime.utcnow().timestamp())}"
    storage_path = deps.note_storage.save(content, note_id)
    existing_ctx = state.get("compound_context") or ""
    return {
        "answer": content,
        "citations": result.citations,
        "retrieved_docs_count": len(result.retrieved_docs),
        "notes_last_output": NoteOutputMeta(
            note_id=note_id, book_title=..., topics=...,
            storage_path=storage_path, created_at=datetime.utcnow().isoformat(),
        ),
        "notes_messages": [AIMessage(content=content)],
        "compound_context": (existing_ctx + f"\n\n[笔记结果]\n{content[:500]}").strip(),
    }
```

**Multi-turn editing logic:**
```
action=new    → retrieve via HybridRetriever → generate fresh note
action=extend → load full note from NoteStorage via storage_path → append new section
action=edit   → load full note from NoteStorage via storage_path → modify specified section
```

If `action=edit/extend` and `storage_path` is `None`, fall back to `action=new`.

**Note templates:**

| Template | Use case | Output structure |
|---|---|---|
| `structured` (default) | Systematic learning | Hierarchical headings + bullets + bold concepts |
| `summary` | Quick review | Core thesis / Evidence / Insight (3 sections) |
| `qa` | Exam prep | Q&A card format |
| `timeline` | History / biography | Chronological nodes |

**Retrieval:** HybridRetriever (BM25 + vector), same config as DeepReadAgent. Constructed once in `__init__`.

**Cross-book synthesis:** When `book_source=None`, retrieve across all books and generate comparative notes.

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
) -> RecommendationResult: ...
```

**Multi-turn awareness:**
Parse `recommend_messages` to extract previously recommended titles → pass as exclusion list to LLM. Detect feedback patterns ("太难了" / "再来几本") and adjust strategy internally.

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
    # One similarity_search per book (generic query, k=2, source filter)
    # Each book → one line: "《title》/ author — snippet[:100]"
    # Cap at 30 books → at most 30 queries total
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
existing_ctx = state.get("compound_context") or ""
return {
    "answer": result.answer,
    "citations": result.citations,
    "retrieved_docs_count": len(result.retrieved_docs),
    "recommend_messages": [AIMessage(content=result.answer)],
    "compound_context": (existing_ctx + f"\n\n[推荐结果]\n{result.answer[:500]}").strip(),
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
    storage_path: str | None = None,
    plan_progress: list[str] | None = None,
) -> ReadingPlanResult: ...
```

`plan_node` follows the same closure pattern as `notes_node`, reads `storage_path` from `plan_last_output` and `plan_progress` from state.

**Multi-turn plan editing:**
```
action=new    → generate fresh plan → save via PlanStorage
action=edit   → load plan from PlanStorage via storage_path → modify → save new version
action=extend → load plan → add books or extend timeline → save new version
```

If `action=edit/extend` and `storage_path` is `None`, fall back to `action=new`.

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
`plan_progress` is a `list[str]` of completed section identifiers. When `action=new/edit/extend`, the agent filters out completed sections from the schedule. When the supervisor detects `is_progress_update=True`, it routes to the plan agent which extracts the section name from user input and appends it to `plan_progress` in the returned delta.

**Node writes back to state:**
```python
# Inside plan_node (module-level, called by _plan closure):
existing_ctx = state.get("compound_context") or ""
return {
    "answer": content,
    "citations": result.citations,
    "retrieved_docs_count": len(result.retrieved_docs),
    "plan_last_output": PlanOutputMeta(
        plan_id=plan_id, book_titles=..., plan_type=plan_type,
        storage_path=storage_path, created_at=datetime.utcnow().isoformat(),
        progress_summary=f"{len(plan_progress or [])} sections completed",
    ),
    "plan_messages": [AIMessage(content=content)],
    "plan_progress": newly_completed_sections or [],   # append reducer
    "compound_context": (existing_ctx + f"\n\n[计划结果]\n{content[:500]}").strip(),
}
```

---

## Section 5: Multi-Agent Pipeline

### Supported compound scenarios

| User says | compound_intents | Notes |
|---|---|---|
| "推荐一本康德的书并生成读书计划" | `["recommend", "plan"]` | recommend output → injected as context into plan task |
| "精读这一章并整理成笔记" | `["deepread", "notes"]` | deepread output → injected into notes task |
| "推荐几本书，挑一本做精读" | `["recommend", "deepread"]` | recommend narrows scope; deepread uses context |
| "把精读做成笔记再更新阅读计划" | `["deepread", "notes", "plan"]` | three-step chain |
| "把这次精读做成笔记"（精读已完成） | `[]` (single: notes) | deepread output in `agent_last_turns` → injected automatically |

### How context flows between agents

```
Step 1: Supervisor dispatches recommend
  pending_agents = ["plan"]
  compound_context = None

Step 2: recommend node runs
  answer = "推荐《判断力批判》，因为..."
  compound_context = "[推荐结果]\n推荐《判断力批判》，因为..."

Step 3: Supervisor dispatches plan
  pending_agents = []
  task_msg = "生成读书计划\n\n【前序步骤结果，供参考】：\n[推荐结果]\n推荐《判断力批判》..."
  plan_messages += [HumanMessage(task_msg)]

Step 4: plan node runs with full context
  → Generates plan specifically for 《判断力批判》

Step 5: finalize
  → Synthesizes both outputs into a single final answer
```

### Unsupported: dynamic fan-out

Requests like "推荐三本书，每本都做笔记" require N note operations for N recommended books. This is not supported — the Supervisor responds: "我可以先推荐，然后你告诉我要对哪本书做笔记。" (`compound_intents` left empty, `intent = "recommend"`).

### Finalize: synthesizing compound results

When `pending_agents` is empty and multiple agents have run, `_finalize` uses `compound_context` to synthesize a unified response rather than just returning the last agent's answer:

```python
def _finalize(state: GraphState) -> dict:
    patch = {"next": "end"}
    answer = state.get("answer") or ""
    ctx = state.get("compound_context") or ""
    # If multiple agents ran, synthesize a unified response
    if ctx and ctx.count("[") > 1:   # more than one agent result in context
        answer = _synthesize_compound_answer(state["user_input"], ctx, answer)
        patch["answer"] = answer
    if answer:
        patch["messages"] = [AIMessage(content=answer)]
    if deps.mem0 and answer and state.get("user_input"):
        deps.mem0.add_qa(state["user_input"], answer)
    return patch
```

---

## Section 6: Storage Layer

### NoteStorage

```python
# backend/storage/note_storage.py

class NoteStorage(Protocol):
    def save(self, content: str, note_id: str) -> str: ...   # returns storage_path
    def load(self, storage_path: str) -> str: ...
    def list(self, prefix: str = "") -> list[str]: ...
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

**Dependency note:** Add `langgraph-checkpoint-sqlite` to `requirements.txt`. Verify the exact API (`from_conn_string` vs constructor) against the pinned LangGraph version before implementation.

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
| `backend/agents/orchestrator_agent.py` | GraphState: 8 new fields (messages×4, pending_agents, compound_context, action, metadata pointers, plan_progress); IntentSchema: compound_intents replaces source_agent; classify_intent: new signature + prompt; Supervisor: pending_agents queue, compound_context injection, delta-return; finalize: compound synthesis; SqliteSaver; GraphDeps adds storage |
| `backend/agents/note_agent.py` | Accept notes_messages + action + notes_format + storage_path; HybridRetriever in __init__; action mode branching; NoteStorage integration in notes_node; writes compound_context |
| `backend/agents/recommendation_agent.py` | Accept recommend_messages + recommend_type; HybridRetriever in __init__; catalog cache; repeat-avoidance; Mem0 preference injection; writes compound_context |
| `backend/agents/reading_plan_agent.py` | Accept plan_messages + action + plan_type + storage_path + plan_progress; HybridRetriever in __init__; chapter extraction; time estimation; action mode branching; PlanStorage integration; writes compound_context |
| `backend/agents/deepread_agent.py` | deepread_node writes compound_context to returned delta |
| `backend/storage/note_storage.py` | New: NoteStorage Protocol + LocalNoteStorage + S3NoteStorage stub |
| `backend/storage/plan_storage.py` | New: PlanStorage Protocol + LocalPlanStorage + S3PlanStorage stub |
| `backend/storage/__init__.py` | New |
| `backend/config.py` | Add NOTE_STORAGE_DIR, PLAN_STORAGE_DIR |
| `requirements.txt` | Add langgraph-checkpoint-sqlite |
