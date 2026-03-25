# Agent Team Architecture Redesign

**Date:** 2026-03-25
**Status:** Draft v2
**Revision:** v3 — Addresses C1–C4, H1–H3, M1–M4 from review 1; N1–N3 from review 2

---

## Background and Goals

The current Kant orchestration layer uses LangGraph with a supervisor ReAct pattern. All agent instances are created once at startup and shared via `GraphDeps`, causing:

- Concurrent requests sharing agent instances (BM25 cache races, state pollution)
- Compound intents execute serially; no intra-request parallelism
- Dependency handling hard-coded in the supervisor prompt
- LangGraph graph structure tightly coupled to business logic

**Goal:** Migrate to a persistent Agent Team pattern using `threading.Queue` for communication, inspired by `examples/agents/s09–s12`. Support explicit dependency graphs and stage-level parallelism within a single request.

**Scope:** This redesign targets single-user or low-concurrency personal use (matching the existing deployment model). Cross-request concurrency is addressed via Worker pools (see C1 below) but is not the primary design driver.

---

## Architecture Overview

```
FastAPI lifespan startup
    └─ AgentTeam.startup()
           ├─ WorkerPool("deepread",  size=2)   # N Worker threads per type
           ├─ WorkerPool("notes",     size=2)
           ├─ WorkerPool("plan",      size=1)   # PlanEditor is stateful; pool=1
           └─ WorkerPool("recommend", size=2)

POST /chat
    └─ Dispatcher.dispatch()
           1. InputSafetyFilter
           2. Mem0 memory search
           3. LLM intent classification → IntentGraph(stages)
           4. For each stage:
              - Dispatch all intents in stage in parallel to WorkerPools
              - Collect all responses (timeout 30s)
              - Extract context, inject into next stage payloads
           5. Aggregate → HTTP Response
```

---

## Component Design

### 1. MessageEnvelope

```python
@dataclass
class MessageEnvelope:
    msg_type: str        # see table below
    request_id: str      # full uuid4().hex — unique per envelope
    sender: str          # "dispatcher" | worker name
    payload: dict        # task params or result
    reply_to: queue.Queue | None  # carried by task_request; worker puts reply here
```

**Message types (extended from s09 VALID_MSG_TYPES):**

| Type | Direction | Description |
|---|---|---|
| `task_request` | Dispatcher → Worker | Assign task, carries `reply_to` |
| `task_response` | Worker → Dispatcher | Task complete, carries result |
| `task_error` | Worker → Dispatcher | Task failed, carries error string |
| `shutdown_request` | AgentTeam → Worker | Request graceful shutdown (s10 FSM) |
| `shutdown_response` | Worker → AgentTeam | Confirm shutdown |
| `broadcast` | AgentTeam → all | Broadcast to all workers |

---

### 2. AgentWorker Base Class

Each Worker is a `daemon=True` `threading.Thread` holding:

- `self.inbox: queue.Queue` — receives messages
- `self._histories: dict[str, list[dict]]` — per-`thread_id` message history (replaces `GraphState` per-agent histories; keyed by `thread_id` to prevent cross-user contamination)
- `self.status: str` — `idle | working | shutdown`
- `self._shutdown_event: threading.Event`

**Run loop:**

```python
def run(self):
    while not self._shutdown_event.is_set():
        try:
            msg = self.inbox.get(timeout=1.0)
        except queue.Empty:
            continue

        if msg.msg_type == "task_request":
            self.status = "working"
            try:
                result = self._execute(msg.payload)
                msg.reply_to.put(MessageEnvelope(
                    msg_type="task_response",
                    request_id=msg.request_id,
                    sender=self.name,
                    payload=result,
                    reply_to=None,
                ))
            except Exception as e:
                msg.reply_to.put(MessageEnvelope(
                    msg_type="task_error",
                    request_id=msg.request_id,
                    sender=self.name,
                    payload={"error": str(e)},
                    reply_to=None,
                ))
            finally:
                self.status = "idle"

        elif msg.msg_type == "shutdown_request":
            # Respond immediately; current task (if any) already completed
            # because inbox.get() only runs when the worker is between tasks.
            self._shutdown_event.set()
            self.status = "shutdown"
            if msg.reply_to:
                msg.reply_to.put(MessageEnvelope(
                    msg_type="shutdown_response",
                    request_id=msg.request_id,
                    sender=self.name,
                    payload={"approve": True},
                    reply_to=None,
                ))
```

Subclasses implement `_execute(payload: dict) -> dict` only.

**Multi-turn history:**
`self._histories[thread_id]` accumulates per session. History is capped at the last 20 messages per thread to prevent context explosion. Threads not seen for 24 hours are evicted from the dict.

---

### 3. WorkerPool (fixes C1)

Each agent type has a pool of N Worker threads. The Dispatcher picks the least-busy Worker. The pool takes a `worker_factory: Callable[[], AgentWorker]` — a factory that returns a fully-constructed Worker (not a bare agent), so multi-argument constructors like `DeepReadWorker(deepread_agent, note_agent)` are handled naturally via `partial` or lambda (fixes N1):

```python
class WorkerPool:
    def __init__(self, name: str, worker_factory: Callable[[], AgentWorker], size: int = 2):
        self.name = name
        self.workers: list[AgentWorker] = [worker_factory() for _ in range(size)]

    def least_busy(self) -> AgentWorker:
        """Return the first idle worker, or the worker with the shortest queue."""
        idle = [w for w in self.workers if w.status == "idle"]
        if idle:
            return idle[0]
        return min(self.workers, key=lambda w: w.inbox.qsize())

    def all_workers(self) -> list[AgentWorker]:
        return self.workers
```

**`AgentWorker` base class exposes `invalidate_bm25()` for cache management (fixes N2):**

```python
class AgentWorker(threading.Thread):
    ...
    def invalidate_bm25(self):
        """Override in subclasses that hold a retriever with a BM25 cache."""
        pass   # no-op by default; not a silent failure
```

Subclasses that wrap agents with retrievers (DeepReadWorker, NotesWorker) override this:
```python
class DeepReadWorker(AgentWorker):
    def invalidate_bm25(self):
        self._deepread_agent._retriever.invalidate_bm25()
```

`AgentTeam.invalidate_bm25_caches()` then calls this method on all workers — no `hasattr` guessing:
```python
def invalidate_bm25_caches(self):
    for pool in self.pools.values():
        for worker in pool.all_workers():
            worker.invalidate_bm25()   # no-op on workers without retrievers
```

Pool sizes:
- `deepread`, `notes`, `recommend`: default 2 (stateless between requests)
- `plan`: 1 (PlanEditor reads/writes plan files by book_id; concurrent edits to the same book are serialized naturally via single-worker queue)

---

### 4. NoteAgent — Two Interfaces (fixes H2)

`NoteAgent` has two distinct uses in the current system:

1. **Auto-hook** (`process_qa`): called automatically after every `deepread` result to silently append Q&A to notes
2. **User-initiated** (`run`): structured notes in 4 formats (`structured/summary/qa/timeline`) with `new/edit/extend` actions

In the new design:

- **Auto-hook** moves inside `DeepReadWorker._execute()`: after `DeepReadAgent.run()` completes, call `note_agent.process_qa(...)` directly (no message passing needed — same thread can call it synchronously)
- **User-initiated notes** is handled by `NotesWorker` which wraps `NoteAgent.run()`

```python
class DeepReadWorker(AgentWorker):
    def __init__(self, deepread_agent, note_agent):
        ...
        self._note_agent = note_agent   # for auto-hook only

    def _execute(self, payload):
        result = self._deepread_agent.run(...)
        # auto-hook: same as current orchestrator behavior
        self._note_agent.process_qa(result.answer, result.citations, ...)
        return {...}
```

`NotesWorker` holds its own independent `NoteAgent` instance for user-initiated requests.

---

### 5. PlanEditor — Dual Use (fixes H3)

`PlanEditor` is used in two places:

1. **`/reader/{book_id}/init`** (REST): calls `PlanEditor.generate()` — non-ReAct, no chat context
2. **Chat** (`plan` intent): calls `PlanEditor.run()` — ReAct agent

In the new design:

- `AgentTeam` owns a `PlanEditor` singleton
- `reader.py` accesses it via `get_plan_editor()` (module-level accessor, similar to existing catalog factory pattern)
- `PlanWorker` wraps the same instance

```python
# backend/team/team.py
_plan_editor: PlanEditor | None = None

def get_plan_editor() -> PlanEditor:
    if _plan_editor is None:
        raise RuntimeError("AgentTeam has not been started — call startup() first")
    return _plan_editor   # set during AgentTeam.startup() (fixes N3)
```

```python
# backend/api/reader.py (small addition)
from backend.team.team import get_plan_editor

@router.post("/reader/{book_id}/init")
async def init_reader(book_id: str):
    plan_editor = get_plan_editor()
    ...
```

`reader.py` is added to the **修改** list.

---

### 6. `invalidate_bm25_caches()` Migration (fixes C4)

Currently `backend/api/books.py` imports:
```python
from backend.agents.orchestrator_agent import invalidate_bm25_caches
```

After `orchestrator_agent.py` is deleted this import breaks. Solution:

```python
# backend/team/team.py
class AgentTeam:
    def invalidate_bm25_caches(self):
        """Called by POST /books/upload after successful ingest."""
        for pool in self.pools.values():
            for worker in pool.all_workers():
                worker.invalidate_bm25()   # polymorphic; no-op on workers without caches
```

`books.py` is added to the **修改** list:
```python
# backend/api/books.py
from backend.team.team import get_agent_team

# after successful ingest:
get_agent_team().invalidate_bm25_caches()
```

---

### 7. `active_tab` Direct Routing (fixes C3)

`ChatRequest` retains `active_tab: str | None`. When set, `Dispatcher` skips LLM intent classification and forces:

```python
if req.active_tab:
    intent_graph = IntentGraph(stages=[[req.active_tab]])
else:
    intent_graph = await self._classify(query, book_meta)
```

---

### 8. IntentGraph and Stage Execution

**Intent classification returns a staged graph:**

```python
@dataclass
class IntentGraph:
    stages: list[list[str]]
    # [["deepread", "recommend"]]       — parallel, no dependency
    # [["recommend"], ["plan"]]          — sequential: plan needs recommend output
    # [["recommend"], ["notes", "plan"]] — recommend first, then notes+plan parallel
```

**Known dependency rules:**

| Downstream | Depends on | Injected field |
|---|---|---|
| `plan` | `recommend` | `book_source` (extracted from `《书名》` in answer) |
| `notes` | `recommend` | `book_source` (same) |

**Stage execution loop:**

```python
context: dict = {}   # upstream outputs, keyed by intent name

for stage in intent_graph.stages:
    reply_queue: queue.Queue = queue.Queue()
    requests: dict[str, str] = {}   # request_id → intent

    for intent in stage:
        req_id = uuid.uuid4().hex        # full 32-char UUID (not truncated)
        requests[req_id] = intent
        payload = build_payload(intent, query, book_meta,
                                memory_ctx, thread_id, upstream=context)
        pool = team.pools[intent]
        pool.least_busy().inbox.put(MessageEnvelope(
            msg_type="task_request", request_id=req_id,
            sender="dispatcher", payload=payload, reply_to=reply_queue,
        ))

    responses = collect(reply_queue, expected=requests, timeout=30.0)
    for req_id, resp in responses.items():
        context[requests[req_id]] = resp.payload   # write stage output to context
```

**`collect()` specification (fixes H1):**

```python
def collect(
    reply_queue: queue.Queue,
    expected: dict[str, str],   # request_id → intent
    timeout: float,
) -> dict[str, MessageEnvelope]:
    """
    Block until all expected request_ids are received or timeout expires.
    Returns a dict of request_id → MessageEnvelope (task_response or task_error).
    Missing entries (timeout) are absent from the result dict.
    The caller checks len(result) < len(expected) to detect partial completion.
    """
    responses: dict[str, MessageEnvelope] = {}
    deadline = time.monotonic() + timeout
    while len(responses) < len(expected):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            resp = reply_queue.get(timeout=remaining)
            responses[resp.request_id] = resp
        except queue.Empty:
            break
    return responses
```

On partial timeout: already-received responses are used; missing intents produce a
`"（未完成）"` placeholder in the aggregate answer.

---

### 9. Async/Blocking Safety (fixes M3)

`Dispatcher.dispatch()` calls `collect()` which blocks on `queue.Queue.get()`.
Since `chat()` is an `async def` FastAPI endpoint, blocking the event loop must be avoided.

`collect()` is run in a thread pool executor:

```python
async def dispatch(self, ...) -> DispatchResult:
    ...
    loop = asyncio.get_event_loop()
    responses = await loop.run_in_executor(
        None, collect, reply_queue, requests, 30.0
    )
    ...
```

Alternatively, the chat endpoint can be `def chat(...)` (synchronous), which FastAPI runs in a thread pool automatically. Either approach is acceptable; the implementation should be consistent throughout.

---

### 10. AgentTeam Lifecycle

```python
class AgentTeam:
    def startup(self):
        plan_editor = PlanEditor()
        global _plan_editor
        _plan_editor = plan_editor

        shared_note_agent = NoteAgent()   # one NoteAgent instance for auto-hooks
        self.pools = {
            # worker_factory returns a fully-constructed AgentWorker (fixes N1)
            "deepread":  WorkerPool("deepread",
                             lambda: DeepReadWorker(DeepReadAgent(), shared_note_agent),
                             size=2),
            "notes":     WorkerPool("notes",
                             lambda: NotesWorker(NoteAgent()),
                             size=2),
            "plan":      WorkerPool("plan",
                             lambda: PlanWorker(plan_editor),
                             size=1),
            "recommend": WorkerPool("recommend",
                             lambda: RecommendWorker(RecommendationAgent()),
                             size=2),
        }
        for pool in self.pools.values():
            for w in pool.all_workers():
                w.start()
        self._write_config("running")

    def shutdown(self):
        """s10 shutdown protocol: broadcast shutdown_request, await responses, join."""
        reply_queue: queue.Queue = queue.Queue()
        total = 0
        for pool in self.pools.values():
            for worker in pool.all_workers():
                req_id = uuid.uuid4().hex
                worker.inbox.put(MessageEnvelope(
                    msg_type="shutdown_request", request_id=req_id,
                    sender="team", payload={}, reply_to=reply_queue,
                ))
                total += 1

        deadline = time.monotonic() + 5.0
        acked = 0
        while acked < total:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                reply_queue.get(timeout=remaining)
                acked += 1
            except queue.Empty:
                break

        for pool in self.pools.values():
            for w in pool.all_workers():
                w.join(timeout=1.0)
        self._write_config("shutdown")
```

**Shutdown race note (M1):** Workers only process inbox messages when between tasks (the `inbox.get()` call blocks inside `run()`). A `shutdown_request` arriving while a Worker is inside `_execute()` will be processed after `_execute()` returns — the current task completes normally and its `task_response` is delivered before the Worker exits. No in-flight request is interrupted.

**Observability (`data/team/config.json`):**
Written at startup and shutdown only (not per status change — avoids high-frequency file I/O). Runtime per-worker status is available via `GET /team/status` (thin endpoint reading `worker.status` from memory):

```json
{
  "status": "running",
  "members": [
    {"name": "deepread-0", "role": "evidence-based Q&A", "status": "idle"},
    {"name": "deepread-1", "role": "evidence-based Q&A", "status": "working"},
    {"name": "plan-0",     "role": "reading plan",        "status": "idle"}
  ]
}
```

---

### 11. FastAPI Integration

```python
# backend/main.py
_team: AgentTeam | None = None

def get_agent_team() -> AgentTeam:
    return _team

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _team
    _team = AgentTeam()
    _team.startup()
    dispatcher = Dispatcher(_team, Mem0Store())
    app.state.dispatcher = dispatcher
    yield
    _team.shutdown()

app = FastAPI(lifespan=lifespan)
```

```python
# backend/api/chat.py
@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    dispatcher: Dispatcher = request.app.state.dispatcher
    result = await dispatcher.dispatch(
        query=req.query,
        book_id=req.book_id,
        thread_id=req.thread_id or "default",
        active_tab=req.active_tab,          # preserved (fixes C3)
        selected_text=req.selected_text,
        current_chapter=req.current_chapter,
    )
    return result.to_dict()
```

---

## File Structure

**New:**

```
backend/team/
    __init__.py
    message.py             # MessageEnvelope, VALID_MSG_TYPES
    worker.py              # AgentWorker base class, WorkerPool
    workers/
        deepread_worker.py
        notes_worker.py
        plan_worker.py
        recommend_worker.py
    team.py                # AgentTeam, get_agent_team(), get_plan_editor()
    dispatcher.py          # Dispatcher, IntentGraph, collect()

data/team/
    config.json            # written at startup/shutdown
```

**Deleted:**

```
backend/agents/orchestrator_agent.py
```

**Modified:**

```
backend/main.py            # lifespan: GraphDeps → AgentTeam + Dispatcher
backend/api/chat.py        # run_minimal_graph() → dispatcher.dispatch()
backend/api/books.py       # invalidate_bm25_caches() → team.invalidate_bm25_caches()
backend/api/reader.py      # PlanEditor access via get_plan_editor()
```

**Unchanged:**

```
backend/agents/deepread_agent.py
backend/agents/note_agent.py
backend/agents/plan_editor.py
backend/agents/recommendation_agent.py
backend/storage/
backend/rag/
backend/memory/
backend/security/input_filter.py
backend/api/notes.py
```

---

## Migration Steps

| Step | Content | Verification |
|---|---|---|
| 1 | `message.py` — MessageEnvelope + types | Unit test: serialize/deserialize |
| 2 | `worker.py` — AgentWorker base + WorkerPool | Mock agent: test task_request, task_error, shutdown FSM |
| 3 | 4 Worker subclasses | Integration test per worker |
| 4 | `team.py` — startup/shutdown + config.json | Test thread start/stop, verify config.json written |
| 5 | `dispatcher.py` — single intent path + active_tab | Wire into chat.py; run `pytest tests/agents/` |
| 6 | `dispatcher.py` — IntentGraph stages + dependency injection | Compound intent integration tests |
| 7 | Migrate books.py and reader.py | Run `pytest tests/api/` |
| 8 | Delete `orchestrator_agent.py` | Full regression: `pytest tests/ -v` |

---

## Correspondence with Examples

| Examples pattern | This design |
|---|---|
| s09 JSONL mailbox + drain-on-read | `threading.Queue` inbox (same-process optimization) |
| s09 TeammateManager + config.json | `AgentTeam` + `data/team/config.json` |
| s09 `_teammate_loop` private messages | `AgentWorker._histories[thread_id]` per-session history |
| s10 shutdown_request/response FSM | `AgentTeam.shutdown()` broadcast + Worker acknowledgement |
| s10 request_id correlation | `MessageEnvelope.request_id` (full uuid4) + `reply_to` Queue |
| s07 task dependency graph | `IntentGraph.stages` staged execution |
| s08 background thread execution | Worker daemon threads in WorkerPool |
