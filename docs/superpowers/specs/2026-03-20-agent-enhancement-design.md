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
| Cross-agent handoff | `handoff_docs` in GraphState (raw Documents) | Information-lossless; no text degradation |
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
  - Write HumanMessage(task) to target agent's messages list
  - Populate handoff_docs if cross-agent pipeline
  ↓
[deepread | notes | plan | recommend] node
  - Read own {agent}_messages as conversation context
  - Use handoff_docs if present (skip retrieval)
  - Generate output
  - Write AIMessage(output) to own {agent}_messages
  - Write {agent}_last_docs for downstream agents
  ↓
Supervisor node  →  detects answer present → finalize
  ↓
finalize node
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

All message lists use LangGraph's `add_messages` reducer and are persisted by SqliteSaver under the same `thread_id`.

---

## Section 2: GraphState Changes

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
    handoff_docs:    list[Document] | None   # raw Documents from previous agent
    handoff_source:  str | None              # "deepread" / "notes" / etc.

    # Last retrieved docs per agent (for downstream handoff)
    deepread_last_docs:   list[Document] | None
    notes_last_docs:      list[Document] | None
    plan_last_docs:       list[Document] | None
    recommend_last_docs:  list[Document] | None

    # ── New: action type ─────────────────────────────────────────────
    action: Literal["new", "edit", "extend"] | None
    # new    → fresh task, no dependency on history
    # edit   → modify a specific part of the last output
    # extend → append to the last output

    # ── New: structured output metadata pointers ─────────────────────
    notes_last_output:   dict | None
    # {note_id, book_title, topics, storage_path, created_at}

    plan_last_output:    dict | None
    # {plan_id, book_titles, plan_type, storage_path, created_at, progress_summary}
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
    source_agent: str | None = None
    # Populated when user implies cross-agent pipeline
    # e.g. "把这次精读做成笔记" → intent=notes, source_agent="deepread"
    notes_format: Literal["structured", "summary", "qa", "timeline"] | None = None
    recommend_type: Literal["discover", "similar", "next", "theme"] | None = None
    plan_type: Literal["single_deep", "multi_theme", "research"] | None = None
```

### Supervisor Routing Logic

```python
def supervisor_node(state: GraphState) -> GraphState:
    # 1. Safety check (unchanged)
    # 2. Intent classification (upgraded IntentSchema)
    result = classify_intent(user_input, agent_histories)
    state["intent"] = result.intent
    state["action"] = result.action

    # 3. answer already present → finalize
    if state.get("answer"):
        state["next"] = "finalize"
        return state

    # 4. Cross-agent pipeline: populate handoff_docs
    if result.source_agent:
        state["handoff_docs"] = state.get(f"{result.source_agent}_last_docs")
        state["handoff_source"] = result.source_agent

    # 5. Dispatch: write HumanMessage to target agent's messages list
    task_msg = HumanMessage(content=build_task_message(user_input, result))
    if result.intent == "notes":
        state["notes_messages"] = [task_msg]   # add_messages accumulates
        state["notes_query"] = user_input
        state["notes_book_source"] = result.book_source or state.get("book_source")
        state["next"] = "notes"
    elif result.intent == "plan":
        state["plan_messages"] = [task_msg]
        state["plan_query"] = user_input
        state["plan_book_source"] = result.book_source or state.get("book_source")
        state["next"] = "plan"
    elif result.intent == "recommend":
        state["recommend_messages"] = [task_msg]
        state["recommend_query"] = user_input
        state["next"] = "recommend"
    else:  # deepread
        state["deepread_messages"] = [task_msg]
        state["deepread_query"] = user_input
        state["deepread_book_source"] = result.book_source or state.get("book_source")
        state["next"] = "deepread"

    return state
```

### finalize Node Upgrade

```python
def finalize(state: GraphState) -> GraphState:
    # Write final answer back into global messages (Supervisor ↔ User)
    if state.get("answer"):
        state["messages"] = [AIMessage(content=state["answer"])]
    # Extract user preferences into Mem0 (unchanged)
    if deps.mem0 and state.get("answer") and state.get("user_input"):
        deps.mem0.add_qa(state["user_input"], state["answer"])
    state["next"] = "end"
    return state
```

---

## Section 4: Agent Enhancements

### 4.1 NoteAgent

**New constructor parameters:**
- `note_storage: NoteStorage` — injected via GraphDeps

**New `run()` parameters:**
- `notes_messages: list[AnyMessage]` — full conversation history
- `action: Literal["new", "edit", "extend"]` — mode
- `notes_format: str` — output template
- `handoff_docs: list[Document] | None` — from cross-agent pipeline

**Multi-turn editing logic:**
```
action=new    → retrieve via HybridRetriever → generate fresh note
action=extend → load previous note from NoteStorage → append new section
action=edit   → load previous note → modify specified section only
```

**Note templates:**

| Template | Use case | Output structure |
|---|---|---|
| `structured` (default) | Systematic learning | Hierarchical headings + bullets + bold concepts |
| `summary` | Quick review | Core thesis / Evidence / Insight (3 sections) |
| `qa` | Exam prep | Q&A card format |
| `timeline` | History / biography | Chronological nodes |

**Retrieval upgrade:** HybridRetriever (BM25 + vector, same as DeepReadAgent).

**Cross-book synthesis:** When `book_source=None`, retrieve across all books and generate comparative notes (e.g. "how different books treat the concept of time").

**NoteStorage integration:**
- After generating note: `storage_path = note_storage.save(content, note_id)`
- Write `notes_last_output = {note_id, book_title, topics, storage_path, created_at}` to state
- Write `AIMessage(note_content)` to `notes_messages`

---

### 4.2 RecommendationAgent

**New `run()` parameters:**
- `recommend_messages: list[AnyMessage]` — full conversation history
- `recommend_type: str` — recommendation sub-type

**Multi-turn awareness:**
- Parse `recommend_messages` to extract previously recommended titles → exclude from next round
- Detect user feedback patterns ("太难了" → filter by difficulty, "再来几本" → same anchor)

**Global book catalog view:**
- Call `list_sources()` to get all available books upfront
- For each book, retrieve 2-3 representative chunks to build a lightweight catalog summary
- Match against user intent from full catalog, not just chunk similarity hits

**Mem0 preference injection:**
Explicitly parse `memory_context` into structured profile:
```
Known preferences: [philosophy, moderate reading pace, accessible writing style]
Previously recommended: [《纯粹理性批判》, 《判断力批判》]
→ Recommend adjacent, less-known works; avoid repeats
```

**Recommendation sub-types:**

| Type | Example trigger | Strategy |
|---|---|---|
| `discover` | "推荐几本小众好书" | Broad catalog scan |
| `similar` | "像这本一样的" | Current book as anchor for similarity |
| `next` | "读完这本读什么" | Difficulty progression or thematic extension |
| `theme` | "关于时间哲学的书" | Topic-focused retrieval across full catalog |

---

### 4.3 ReadingPlanAgent

**New constructor parameters:**
- `plan_storage: PlanStorage` — injected via GraphDeps

**New `run()` parameters:**
- `plan_messages: list[AnyMessage]` — full conversation history
- `action: Literal["new", "edit", "extend"]` — mode
- `plan_type: str` — plan sub-type

**Multi-turn plan editing:**
```
action=new    → generate fresh plan, save via PlanStorage
action=edit   → load existing plan from PlanStorage → modify specified part → save new version
action=extend → load existing plan → add new books or extend timeline
```

**Real chapter structure extraction:**
Extract from ChromaDB chunk metadata:
- `chapter_title` / `section_title` → build chapter list
- Sum `len(page_content)` per chapter → estimate word count
- Apply reading speed: Chinese ~300 chars/min, English ~200 words/min

Example output:
```
《纯粹理性批判》estimated reading times:
  先验感性论  ~8,000 chars  → ~27 min
  先验分析论  ~45,000 chars → ~2.5 hr
  先验辩证论  ~60,000 chars → ~3.3 hr
  Total: ~6.5 hours
```

**Plan sub-types:**

| Type | Use case | Output structure |
|---|---|---|
| `single_deep` | Deep-read one book | Chapter-by-chapter schedule + reading goals |
| `multi_theme` | Thematic reading list | Cross-book interleaved reading route |
| `research` | Research a topic | Key chapters across books + annotation targets |

**Progress tracking:**
User can say "先验感性论我读完了" → Supervisor writes this as `HumanMessage` to `plan_messages` → Next plan generation skips completed sections.

**PlanStorage integration:**
- After generating plan: `storage_path = plan_storage.save(content, plan_id)`
- Write `plan_last_output = {plan_id, book_titles, plan_type, storage_path, created_at, progress_summary}` to state
- Write `AIMessage(plan_content)` to `plan_messages`

---

## Section 5: Cross-Agent Pipeline

### Trigger examples

| User says | Supervisor detects | Pipeline |
|---|---|---|
| "把这次精读做成笔记" | intent=notes, source_agent="deepread" | DeepRead docs → NoteAgent |
| "根据刚才的推荐制定计划" | intent=plan, source_agent="recommend" | Recommend docs → ReadingPlanAgent |

### Implementation

Each agent node writes its retrieved docs to `{agent}_last_docs` in state:
```python
# In deepread node:
state["deepread_last_docs"] = result.retrieved_docs
```

Supervisor populates handoff when `source_agent` is detected:
```python
state["handoff_docs"] = state.get(f"{result.source_agent}_last_docs")
state["handoff_source"] = result.source_agent
```

Target agent checks `handoff_docs` before retrieval:
```python
def run(self, *, query, handoff_docs=None, ...):
    if handoff_docs:
        docs = handoff_docs   # skip retrieval entirely
    else:
        docs = self.retriever.search(query, filter=filter_)
```

---

## Section 6: Storage Layer

### NoteStorage

```python
# backend/storage/note_storage.py

class NoteStorage(Protocol):
    def save(self, content: str, note_id: str) -> str: ...  # returns storage_path
    def load(self, storage_path: str) -> str: ...

class LocalNoteStorage:
    """Stores notes as .md files under data/notes/"""
    root: Path  # configured via NOTE_STORAGE_DIR in config.py

class S3NoteStorage:
    """Stores notes in S3-compatible object storage"""
    bucket: str
    prefix: str = "notes/"
```

### PlanStorage

```python
# backend/storage/plan_storage.py

class PlanStorage(Protocol):
    def save(self, content: str, plan_id: str) -> str: ...
    def load(self, storage_path: str) -> str: ...

class LocalPlanStorage:
    """Stores plans as .md files under data/plans/"""
    root: Path  # configured via PLAN_STORAGE_DIR in config.py

class S3PlanStorage:
    """Stores plans in S3-compatible object storage"""
    bucket: str
    prefix: str = "plans/"
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
    note_storage:     NoteStorage = field(default_factory=LocalNoteStorage)
    plan_storage:     PlanStorage = field(default_factory=LocalPlanStorage)
```

### Persistence: SqliteSaver

```python
# orchestrator_agent.py — one-line change

# Before:
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# After:
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("data/checkpoints.db")
```

All GraphState fields (including per-agent message lists and last_output metadata) are automatically persisted under `thread_id`. Process restarts do not lose conversation history.

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
| `backend/agents/orchestrator_agent.py` | GraphState: 8 new fields; IntentSchema: 4 new fields; Supervisor routing upgrade; finalize writes to messages; SqliteSaver; GraphDeps adds storage |
| `backend/agents/note_agent.py` | Accept notes_messages + action + notes_format + handoff_docs; HybridRetriever; action mode branching; NoteStorage integration |
| `backend/agents/recommendation_agent.py` | Accept recommend_messages + recommend_type; global catalog view; repeat-avoidance; structured Mem0 preference injection |
| `backend/agents/reading_plan_agent.py` | Accept plan_messages + action + plan_type; chapter extraction; word-count time estimation; action mode branching; PlanStorage integration |
| `backend/storage/note_storage.py` | New file: NoteStorage Protocol + LocalNoteStorage |
| `backend/storage/plan_storage.py` | New file: PlanStorage Protocol + LocalPlanStorage |
| `backend/storage/__init__.py` | New file |
| `backend/config.py` | Add NOTE_STORAGE_DIR, PLAN_STORAGE_DIR |
