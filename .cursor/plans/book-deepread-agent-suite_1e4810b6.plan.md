---
name: book-deepread-agent-suite
overview: 在现有 Kant RAG（OpenAI + Chroma + PDF 入库流水线）基础上，落地“智能小众书籍精读多智能体系统”：提供本地 HTTP API，5 个智能体（总控/推荐/精读/笔记/书单规划）共享状态记忆，所有回答带可追溯引用，并加入基础安全防护与测试/版本管理。
todos:
  - id: api-layer
    content: 引入 FastAPI/Uvicorn；实现 backend/api/app.py、补齐 backend/api/chat.py 与 schemas，使系统可通过本地 HTTP 调用（/chat、/ingest、/notes、/plan、/state）
    status: pending
  - id: agent-core
    content: 规范化 BaseAgent 协议；实现 5 个智能体类与 orchestrator graph（LangGraph StateGraph），完成路由、汇总与状态更新
    status: pending
  - id: memory-xai-security
    content: 实现 SharedMemoryStore 与 ReadingState；补齐 citation 构建与答案格式化；补齐 input_filter/prompt_guard 并在 API 与 graph 中强制执行
    status: pending
  - id: tests-logs
    content: 补齐 agents/security/xai 的单元测试；增加 JSONL 事件日志，形成轻量 LLMSecOps/MLSecOps 闭环
    status: pending
isProject: false
---

## 目标与约束

- **目标**：实现 Book-DeepRead Agent Suite 的端到端闭环：`PDF 入库 → 章节/知识点检索 → 精读总结/笔记整理/书单规划/问答`，对外提供本地 HTTP API。
- **硬约束**：LLM 调用 **OpenAI**（你已封装在 `[backend/llm/openai_client.py](g:\pycharm\Kant\backend\llm\openai_client.py)`），向量库用 **Chroma**（你已实现 `[backend/rag/chroma/chroma_store.py](g:\pycharm\Kant\backend\rag\chroma\chroma_store.py)`）。
- **现状评估**（从代码读到的事实）：
  - RAG 侧已经完整：PDF 抽取/清洗/切块/入库/检索（`ChromaStore`）。
  - Agents 侧大多是空壳：`deepread_agent.py` / `recommendation_agent.py` / `note_agent.py` / `reading_plan_agent.py` 目前只有文件头；`orchestrator_agent.py` 现用 `create_agent(...)` 但缺少 import/定义，且不适合做可控的多智能体编排。
  - 安全与可解释性模块：`[backend/xai/citation.py](g:\pycharm\Kant\backend\xai\citation.py)`、`[backend/security/prompt_guard.py](g:\pycharm\Kant\backend\security\prompt_guard.py)`、`[backend/security/input_filter.py](g:\pycharm\Kant\backend\security\input_filter.py)` 目前是空文件头，需要补齐实现。

## 总体架构（推荐：LangChain + LangGraph StateGraph）

- **编排框架**：继续使用 LangChain 生态，但把多智能体“调度+共享内存”落在 **LangGraph**（属于 LangChain 官方路线）上：
  - 优点：显式状态（阅读偏好/进度/笔记/选书）可审计、可回放；节点就是智能体；可以插入 guardrails（安全过滤、引用强制）。
  - 兼容：你现有 `get_llm()` / `ChromaStore.as_retriever()` 可直接作为节点依赖。

```mermaid
flowchart TD
  client[Client] --> api[FastAPI_API]
  api --> guard[InputFilter_PromptGuard]
  guard --> graph[AgentGraph_Orchestrator]

  graph --> recAgent[RecommendationAgent]
  graph --> deepAgent[DeepReadAgent]
  graph --> noteAgent[NoteAgent]
  graph --> planAgent[ReadingPlanAgent]

  recAgent --> llm[OpenAI_ChatOpenAI]
  deepAgent --> retriever[ChromaStore_Retriever]
  deepAgent --> llm
  noteAgent --> llm
  planAgent --> llm

  deepAgent --> cite[CitationBuilder]
  retriever --> cite
  cite --> response[Response_With_Citations]

  graph --> mem[SharedMemoryStore]
  recAgent --> mem
  deepAgent --> mem
  noteAgent --> mem
  planAgent --> mem
```



## 目录与实现类（按你当前结构落位）

### 1) API 层（新增/补齐）

- 新增依赖：`fastapi`、`uvicorn[standard]`（用于本地 HTTP API）。
- 文件与类：
  - `[backend/api/app.py](g:\pycharm\Kant\backend\api\app.py)`
    - `create_app() -> FastAPI`
    - 注册路由、依赖注入（settings、store、graph、memory、guards）
  - `[backend/api/chat.py](g:\pycharm\Kant\backend\api\chat.py)`（现为空壳，补齐）
    - `POST /chat`：统一入口，交给 orchestrator graph
    - `POST /ingest`：触发 `ChromaStore.ingest_pdf`（指定 pdf 路径或批量 ingest）
    - `GET /library/stats`：`ChromaStore.get_stats()`
    - `GET /library/sources`：`ChromaStore.list_sources()`
    - `POST /notes` / `GET /notes`：写入/读取共享笔记（走 SharedMemoryStore）
    - `GET /state`：读取当前阅读状态（进度/当前书/当前章节等）
  - `[backend/api/schemas.py](g:\pycharm\Kant\backend\api\schemas.py)`
    - Pydantic DTO：`ChatRequest`、`ChatResponse`、`IngestRequest`、`NoteUpsertRequest`、`ReadingStateResponse`、`CitationDTO`

### 2) 智能体层（5 个角色 + 可选 QA）

把现有 `backend/agents/*` 统一成“可测试、可注入依赖”的类，而不是直接写 module-level agent。

- `[backend/agents/base_agent.py](g:\pycharm\Kant\backend\agents\base_agent.py)`
  - 规范化为：`BaseAgent(ABC)`（类名首字母大写）、统一 `run(state, input) -> state_patch` 的协议
  - 定义通用依赖注入：`llm`、`retriever`、`memory`、`guards`
- `[backend/agents/orchestrator_agent.py](g:\pycharm\Kant\backend\agents\orchestrator_agent.py)`
  - 替换当前不完整的 `create_agent(...)` 写法
  - `OrchestratorAgentGraph`（或 `build_agent_graph()`）
    - 负责：意图识别（推荐/精读/笔记/书单/问答）、路由到对应 agent、汇总结果、更新阅读进度
- `[backend/agents/recommendation_agent.py](g:\pycharm\Kant\backend\agents\recommendation_agent.py)`
  - `RecommendationAgent`
    - 输入：用户偏好/当前主题
    - 输出：推荐“本地库”里可能相关的书（基于 `list_sources()` + 元数据/检索）
    - 说明：你选择 `pdf_only`，所以推荐依据主要来自 **Chroma 元数据（pdf_title/pdf_author/source）** + **对全库做主题检索**。
- `[backend/agents/deepread_agent.py](g:\pycharm\Kant\backend\agents\deepread_agent.py)`
  - `DeepReadAgent`
    - 核心：RAG 精读（章节解释、知识点提炼、精读建议）
    - 通过 `ChromaStore.as_retriever(search_kwargs={k, filter})` 检索
    - 强制产出：`answer` + `citations[]`（来自命中的 `Document.metadata`）
- `[backend/agents/note_agent.py](g:\pycharm\Kant\backend\agents\note_agent.py)`
  - `NoteAgent`
    - 负责：笔记结构化（要点、引用页码、TODO、问题列表）、“文字版思维导图”（输出 markdown/缩进树）
    - 写入 SharedMemoryStore：`notes`、`note_index`（便于后续问答/书单参考）
- `[backend/agents/reading_plan_agent.py](g:\pycharm\Kant\backend\agents\reading_plan_agent.py)`
  - `ReadingPlanAgent`
    - 负责：根据目标（社科入门/专业提升）与可用书源（sources）生成计划、节奏与调整策略
    - 输出：`reading_plan`（可序列化 JSON），并写入 memory
- （可选增强，不增人头但提升体验）新增 `[backend/agents/qa_agent.py](g:\pycharm\Kant\backend\agents\qa_agent.py)`
  - `QAAagent` 专职“书中即时问答”，由 orchestrator 路由；也可并入 `DeepReadAgent` 的一个 mode。

### 3) 共享内存与状态（支持“模拟同步阅读进度”）

- 新增 `[backend/memory/state.py](g:\pycharm\Kant\backend\memory\state.py)`
  - `ReadingState`：当前书(source/title)、当前页/章节（可模拟）、偏好标签、历史查询、最近引用
  - `UserProfile`：偏好（领域、作者/译者倾向、难度、篇幅）、禁忌（不偏向畅销书）
- 新增 `[backend/memory/store.py](g:\pycharm\Kant\backend\memory\store.py)`
  - `SharedMemoryStore`
    - 默认内存态（dict）
    - 可选落盘：`data/state.json`（满足你“纯本地、无需复杂监控”）

### 4) XAI/RAI：引用与去幻觉策略

- 补齐/新增：
  - `[backend/xai/citation.py](g:\pycharm\Kant\backend\xai\citation.py)`
    - `Citation` dataclass：`source`, `pdf_title`, `pdf_author`, `page_numbers`, `chunk_id`, `snippet`
    - `CitationBuilder.from_docs(docs) -> list[Citation]`
  - `[backend/xai/answer_formatter.py](g:\pycharm\Kant\backend\xai\answer_formatter.py)`
    - 统一把 `answer + citations` 格式化成 API 输出
- **强制引用**策略（减少幻觉）：
  - DeepRead/QA 的 system prompt：要求每条关键结论必须能对应至少 1 条引用；否则明确返回“不足以从本地书库证据回答”。
  - 若检索为空：直接拒答/建议先 ingest。

### 5) 安全：prompt 注入、防越权、输入过滤

- 补齐：
  - `[backend/security/input_filter.py](g:\pycharm\Kant\backend\security\input_filter.py)`
    - `InputFilter.allow(text) -> (ok, reason)`：简单关键词/模式过滤（如“忽略所有规则/泄露 key/修改数据库”等）
  - `[backend/security/prompt_guard.py](g:\pycharm\Kant\backend\security\prompt_guard.py)`
    - `PromptGuard.sanitize(user_text, state) -> sanitized_text`
    - `PromptGuard.enforce_policies()`：将“只能用本地书库/必须引用/拒绝无关需求”注入到 system prompt
- 约束边界：
  - 知识源只来自 Chroma 检索结果（pdf chunks）
  - API 不提供任意文件读取；`/ingest` 只允许 ingest `data/books/` 下文件（路径白名单）

### 6) 版本管理/测试/日志（MLSecOps/LLMSecOps 的轻量落地）

- 版本：
  - 新增 `data/chroma/`（你已用 persist_dir）+ `data/state.json`（memory 可选落盘）
  - 新增 `data/logs/`：JSONL 日志（查询、写笔记、计划生成）
- 测试：沿用现有 `pytest`
  - 新增 `tests/agents/test_orchestrator_routing.py`：路由正确性（用 monkeypatch mock LLM）
  - 新增 `tests/security/test_input_filter.py`：关键注入/越权输入被拦截
  - 新增 `tests/xai/test_citation_builder.py`：引用字段完整且来自 doc.metadata
- 日志：新增 `[backend/observability/logger.py](g:\pycharm\Kant\backend\observability\logger.py)`
  - `log_event(type, payload)`，写入 `data/logs/events.jsonl`

## 技术选型清单（与你的约束对齐）

- **LLM**：OpenAI（LangChain `ChatOpenAI`，复用 `get_llm()`）
- **Embeddings**：OpenAI Embeddings（复用 `get_embeddings()`）
- **向量库**：Chroma（复用 `ChromaStore`，本地 PersistentClient；可保留 CloudClient 兼容）
- **多智能体编排**：LangGraph（StateGraph）
- **API**：FastAPI + Uvicorn（本地服务）
- **测试**：pytest（你已配置）

## 交付物（你会在仓库里看到什么）

- 一个可运行的本地服务：`python -m backend.api.app` 或 `uvicorn backend.api.app:create_app --factory`
- 5 个智能体类 + 1 个编排图构建器
- 共享状态/笔记/计划的本地持久化（可开关）
- 所有精读/问答输出都包含 citations（source + pages + chunk_id + snippet）
- 基础安全过滤、单元测试与日志落盘

## 与现有代码的关键复用点

- 入库/检索：直接复用 `ChromaStore.ingest_pdf()` 与 `ChromaStore.as_retriever()`（见 `[backend/rag/chroma/chroma_store.py](g:\pycharm\Kant\backend\rag\chroma\chroma_store.py)`）
- LLM/Embeddings：复用 `[backend/llm/openai_client.py](g:\pycharm\Kant\backend\llm\openai_client.py)`
- PDF 流水线：复用 `PDFExtractor`/`TextCleaner`/`TextChunker`

