# Agent Team 架构重设计

**日期：** 2026-03-25
**状态：** Draft
**作者：** brainstorming session

---

## 背景与目标

当前 Kant 项目的多 Agent 编排基于 LangGraph，以单图 + supervisor ReAct 模式运行。所有 Agent 实例在服务启动时创建一次，通过 `GraphDeps` 共享，存在以下问题：

- 并发请求共享 Agent 实例（BM25 缓存竞争、状态污染）
- 复合意图串行执行，无法并行
- 依赖关系硬编码在 supervisor prompt 中，不可观测
- LangGraph 图结构与业务逻辑耦合，扩展困难

**目标：** 参照 `examples/agents/s09-s12` 的 Agent Team 模式，用 `threading.Queue` 作为通信层，重构为持久化 Worker 线程团队，支持并行执行和显式依赖管理。

---

## 架构概览

```
FastAPI lifespan startup
    └─ AgentTeam.startup()
           ├─ DeepReadWorker  (常驻线程，inbox: Queue)
           ├─ NotesWorker     (常驻线程，inbox: Queue)
           ├─ PlanWorker      (常驻线程，inbox: Queue)
           └─ RecommendWorker (常驻线程，inbox: Queue)

POST /chat
    └─ Dispatcher.dispatch()
           1. InputSafetyFilter
           2. Mem0 记忆查询
           3. LLM 意图分类 → IntentGraph(stages)
           4. 按阶段执行：
              - 阶段内：并行投递 MessageEnvelope 到各 Worker.inbox
              - 阶段间：等待完成后提取 context，注入下一阶段 payload
           5. 聚合结果 → HTTP Response
```

---

## 组件设计

### 1. MessageEnvelope

```python
@dataclass
class MessageEnvelope:
    msg_type: str       # 见下方消息类型表
    request_id: str     # uuid4()[:8]，关联请求与响应
    sender: str         # "dispatcher" | worker name
    payload: dict       # 任务参数或执行结果
    reply_to: queue.Queue | None  # task_request 携带，Worker 直接 put 回复
```

**消息类型（扩展自 s09 VALID_MSG_TYPES）：**

| 类型 | 方向 | 说明 |
|---|---|---|
| `task_request` | Dispatcher → Worker | 分配任务，携带 reply_to |
| `task_response` | Worker → Dispatcher | 任务完成，携带结果 |
| `task_error` | Worker → Dispatcher | 任务失败，携带错误信息 |
| `shutdown_request` | AgentTeam → Worker | 请求优雅关闭（s10 FSM） |
| `shutdown_response` | Worker → AgentTeam | 确认关闭 |
| `broadcast` | AgentTeam → all | 全员广播 |

---

### 2. AgentWorker（基类）

每个 Worker 是一个 `daemon=True` 的 `threading.Thread`，持有：
- `self.inbox: queue.Queue` — 接收消息
- `self._messages: list[dict]` — 线程私有的多轮对话历史（仿 s09 `_teammate_loop`）
- `self.status: str` — `idle | working | shutdown`
- `self._shutdown_event: threading.Event`

**运行循环：**

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
                    msg_type="task_response", request_id=msg.request_id,
                    sender=self.name, payload=result, reply_to=None,
                ))
            except Exception as e:
                msg.reply_to.put(MessageEnvelope(
                    msg_type="task_error", request_id=msg.request_id,
                    sender=self.name, payload={"error": str(e)}, reply_to=None,
                ))
            finally:
                self.status = "idle"
        elif msg.msg_type == "shutdown_request":
            self._shutdown_event.set()
            self.status = "shutdown"
            msg.reply_to.put(MessageEnvelope(
                msg_type="shutdown_response", request_id=msg.request_id,
                sender=self.name, payload={"approve": True}, reply_to=None,
            ))
```

子类只需实现 `_execute(payload: dict) -> dict`。

**多轮历史说明：**
`self._messages` 在 Worker 生命周期内持续累积，不跨用户隔离（默认行为）。如需按用户隔离，改为 `dict[thread_id, list]`，视后续需求决定。历史超过 N 条时做末尾截断，防止上下文爆炸。

---

### 3. IntentGraph 与分阶段执行

**意图分类返回结构：**

```python
@dataclass
class IntentGraph:
    stages: list[list[str]]
    # 示例：
    # [["deepread", "recommend"]]    → 并行，无依赖
    # [["recommend"], ["plan"]]      → 串行，plan 依赖 recommend
    # [["recommend"], ["notes","plan"]] → recommend 完成后，notes/plan 并行
```

**已知依赖规则：**

| 下游 Agent | 依赖 | 注入字段 |
|---|---|---|
| `plan` | `recommend` | `book_source`（从 recommend 输出提取 `《书名》`） |
| `notes` | `recommend` | `book_source`（同上） |

**分阶段执行逻辑：**

```python
context: dict = {}

for stage in intent_graph.stages:
    reply_queue = queue.Queue()
    requests = {}

    for intent in stage:
        req_id = uuid.uuid4().hex[:8]
        requests[req_id] = intent
        payload = build_payload(intent, query, book_meta, memory_ctx,
                                upstream=context)
        team.workers[intent].inbox.put(
            MessageEnvelope(msg_type="task_request", request_id=req_id,
                            sender="dispatcher", payload=payload,
                            reply_to=reply_queue)
        )

    # 等待本阶段全部响应（30s 超时）
    responses = collect(reply_queue, count=len(requests), timeout=30.0)

    # 上游输出写入 context，供下阶段注入
    for req_id, resp in responses.items():
        context[requests[req_id]] = resp.payload
```

**执行时间对比：**

| IntentGraph | 总耗时 |
|---|---|
| `[["deepread","recommend"]]` | `max(deepread, recommend)` |
| `[["recommend"],["plan"]]` | `recommend + plan` |
| `[["recommend"],["notes","plan"]]` | `recommend + max(notes, plan)` |

---

### 4. AgentTeam 生命周期

```python
class AgentTeam:
    def startup(self):
        self.workers = {
            "deepread":  DeepReadWorker(DeepReadAgent()),
            "notes":     NotesWorker(NoteAgent()),
            "plan":      PlanWorker(PlanEditor()),
            "recommend": RecommendWorker(RecommendationAgent()),
        }
        for w in self.workers.values():
            w.start()
        self._save_config("running")   # data/team/config.json

    def shutdown(self):
        # s10 shutdown 协议：广播 shutdown_request，等待响应，join
        reply_queue = queue.Queue()
        for name, worker in self.workers.items():
            req_id = uuid.uuid4().hex[:8]
            worker.inbox.put(MessageEnvelope(
                msg_type="shutdown_request", request_id=req_id,
                sender="team", payload={}, reply_to=reply_queue,
            ))
        # 5s 超时后强制 join
        deadline = time.monotonic() + 5.0
        acked = 0
        while acked < len(self.workers):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                reply_queue.get(timeout=remaining)
                acked += 1
            except queue.Empty:
                break
        for w in self.workers.values():
            w.join(timeout=1.0)
        self._save_config("shutdown")
```

**可观测性文件（`data/team/config.json`）：**

```json
{
  "status": "running",
  "members": [
    {"name": "deepread",  "role": "evidence-based Q&A", "status": "idle"},
    {"name": "notes",     "role": "note taking",         "status": "working"},
    {"name": "plan",      "role": "reading plan",        "status": "idle"},
    {"name": "recommend", "role": "book recommendation", "status": "idle"}
  ]
}
```

---

### 5. FastAPI 集成

```python
# backend/main.py

@asynccontextmanager
async def lifespan(app: FastAPI):
    team.startup()
    yield
    team.shutdown()

app = FastAPI(lifespan=lifespan)
```

```python
# backend/api/chat.py（改动极小）

@router.post("/chat")
async def chat(req: ChatRequest):
    result = await dispatcher.dispatch(
        query=req.query,
        book_id=req.book_id,
        thread_id=req.thread_id or "default",
        selected_text=req.selected_text,
        current_chapter=req.current_chapter,
    )
    return result.to_dict()
```

---

## 文件结构

**新增：**

```
backend/team/
    __init__.py
    message.py           # MessageEnvelope, VALID_MSG_TYPES
    worker.py            # AgentWorker 基类
    workers/
        deepread_worker.py
        notes_worker.py
        plan_worker.py
        recommend_worker.py
    team.py              # AgentTeam（lifecycle + config.json）
    dispatcher.py        # Dispatcher（分类、分阶段执行、聚合）

data/team/
    config.json          # 运行时生成
```

**删除：**

```
backend/agents/orchestrator_agent.py
```

**修改：**

```
backend/main.py          # lifespan：GraphDeps → AgentTeam + Dispatcher
backend/api/chat.py      # run_minimal_graph() → dispatcher.dispatch()
```

**不动：**

```
backend/agents/deepread_agent.py
backend/agents/note_agent.py
backend/agents/plan_editor.py
backend/agents/recommendation_agent.py
backend/storage/
backend/rag/
backend/memory/
backend/security/input_filter.py
backend/api/reader.py
backend/api/notes.py
```

---

## 迁移顺序

| 步骤 | 内容 | 可测试点 |
|---|---|---|
| 1 | `message.py` | 单元测试 MessageEnvelope 序列化/反序列化 |
| 2 | `worker.py` 基类 | Mock agent，测试 task_request / shutdown FSM |
| 3 | 4 个 Worker 子类 | 各自集成测试 |
| 4 | `team.py` | 测试线程启停、config.json 写入 |
| 5 | `dispatcher.py` 单意图路径 | 替换 chat.py，跑现有 e2e 测试 |
| 6 | `dispatcher.py` 分阶段 + 依赖注入 | 复合意图测试用例 |
| 7 | 删除 `orchestrator_agent.py` | 全量回归测试 |

---

## 不变的层

- 所有 Agent 内部逻辑（DeepReadAgent、NoteAgent、PlanEditor、RecommendationAgent）
- Mem0 长期记忆层
- InputSafetyFilter（移入 Dispatcher，逻辑不变）
- FastAPI 路由与请求/响应 schema（前端零感知）
- SQLite catalog、ChromaDB、BM25 检索层

---

## 与 examples 的对应关系

| Examples 模式 | 本设计映射 |
|---|---|
| s09 JSONL mailbox + drain-on-read | `threading.Queue` inbox（同进程优化版） |
| s09 TeammateManager + config.json | `AgentTeam` + `data/team/config.json` |
| s09 `_teammate_loop` 私有 messages | `AgentWorker.self._messages` 线程私有历史 |
| s10 shutdown_request/response FSM | `AgentTeam.shutdown()` 广播 + Worker 响应 |
| s10 request_id 关联模式 | `MessageEnvelope.request_id` + `reply_to` Queue |
| s07 任务依赖图 | `IntentGraph.stages` 分阶段执行 |
| s08 后台线程执行 | Worker 常驻 daemon 线程 |
