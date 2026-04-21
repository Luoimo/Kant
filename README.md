# Kant — 读书 AI 助手 (Agentic AI System)

**Kant** 是一个基于 **Agentic AI** 架构设计的深度阅读系统。本项目作为 **AAS Practice Module** 的落地实践，旨在超越单一的“AI 功能”调用，构建一个具备多智能体协作、安全可靠、且具备完整工程化落地的 **Agentic AI 系统**。

通过上传 EPUB 书籍，Kant 能够为用户提供沉浸式的精读问答、结构化的笔记整理、个性化的书单推荐以及科学的阅读计划管理，同时支持多轮对话与长期记忆。

---

## 核心能力与项目特色 (AAS 模块实践)

本项目严格遵循项目要求，全面覆盖以下四大核心能力：

### 1. Agentic AI 系统设计 (多智能体协作)
系统采用了清晰的 **Router + 专家/评估/后处理 Agent** 架构编排，确保阅读问答场景的高效与严谨。
- **明确分工**：拥有五个专门的执行 Agent：前端路由分类（RouterAgent）、精读问答检索（DeepReadAgent）、后置客观性与幻觉评估（CriticAgent）、笔记提炼整理（NoteAgent）、以及引导思考（FollowupAgent）。
- **通信与协调**：采用经典的 Agent 协作模式（Supervisor/Critic 机制与 Hook 机制）。例如，用户提问首先由 RouterAgent 进行意图识别和查询重写优化，然后由 DeepReadAgent 基于 RAG 进行回答，回答结束后同步触发 CriticAgent 进行事实核查，同时触发 NoteAgent 提炼笔记、FollowupAgent 生成追问。
- **状态管理**：结合 **Mem0** 实现跨会话的用户长期记忆（阅读偏好、知识背景等）。

### 2. AI Security (大模型安全防御)
全面识别并缓解大模型应用中的核心风险：
- **Prompt Injection 与 Adversarial Inputs**：接入 **Lakera Guard** 企业级安全网关（并在本地内置正则兜底规则），有效拦截越狱（Jailbreak）、提示词注入以及恶意系统指令。
- **敏感信息泄露与系统越权**：拦截尝试在对话中输出 API Key/Token 的行为，严格限制文件系统访问和代码执行请求。
- **Hallucination（幻觉控制）**：采用高级 **RAG 混合检索架构**（BM25 + ChromaDB 向量 + RRF 融合 + LLM Reranker），强制 Agent **严格基于检索到的原文片段**进行回答，大幅降低幻觉。

### 3. Explainable & Responsible AI (可解释性与负责任的 AI)
- **Explainability (可解释性)**：所有来自精读的回答都会附带精确的 **Citations (引用来源)**，并在前端 UI 清晰标明原文出处（书名与章节片段），确保 AI 的回答过程透明、可追溯。
- **Governance & Off-topic Control (治理与控制)**：通过输入过滤器实现“软警告”机制，当用户讨论与“阅读/精读”完全无关的话题时，系统会温和提示，确保 AI 助手保持其核心用途。

### 4. MLSecOps / LLMSecOps (系统集成与自动化部署)
具备完整的现代工程化开发流水线：
- **CI/CD 流水线**：通过 GitHub Actions 配置了自动化的构建与测试流程。
- **自动化测试**：拥有近 200 个单元与集成测试（Pytest），覆盖核心逻辑。
- **AI 安全测试 (Security Tests)**：单独设立安全测试模块，每次提交自动验证防御机制（Prompt Injection、文件访问等）是否依然生效，防止安全规则退化。
- **系统架构设计**：Vue.js / Vite 前端配合 FastAPI 高性能异步后端，采用高度模块化的数据流设计（SQLite 管理元数据，ChromaDB 管理向量，本地存储管理 Markdown 笔记）。

---

## 架构概览

```
EPUB → EpubExtractor → TextCleaner → TextChunker → OpenAI Embeddings → ChromaDB
                                                                           ↓
用户问题 → FastAPI /chat → RouterAgent（意图分类 & 查询优化）
                               ├─ Mem0 长期记忆检索
                               ├─ InputSafetyFilter（每轮安全检查）
                               └─ 核心问答流：
                                   ├─ DeepReadAgent（主流程：基于 RAG 检索回答，附带证据引用）
                                   ├─ CriticAgent（评估流：同步进行事实核查与客观性审查）
                                   ├─ NoteAgent（后处理：自动提炼问答生成结构化笔记并持久化）
                                   └─ FollowupAgent（后处理：根据对话生成深入追问问题）
```

---

## 目录结构

```
backend/
  agents/
    router_agent.py        前端路由 Agent，负责意图分类和查询重写优化
    deepread_agent.py      精读问答 Agent，基于检索内容生成带引用的回答
    critic_agent.py        评论员 Agent，事实核查与客观性评估（防止幻觉）
    note_agent.py          笔记整理 Agent，将问答提炼为结构化笔记并持久化
    followup_agent.py      追问 Agent，在一轮问答结束后生成相关引导问题
  api/
    chat.py                POST /chat, POST /books/upload, GET /books
    reader.py              Reader Mode 端点（init/plan/progress）
    notes.py               笔记端点
    books.py               书籍管理端点
  config.py                Pydantic Settings，读取 .env
  llm/                     OpenAI LLM & Embeddings 封装
  main.py                  FastAPI app 入口
  memory/
    mem0_store.py          Mem0 长期记忆封装
  rag/
    chroma/
      chroma_store.py      ChromaStore（向量库管理，支持本地/Cloud 双模式）
    chunker/               文本切块（TextChunker）
    cleaner/               文本清洗（TextCleaner）
    extracter/             EPUB 解析（EpubExtractor，含封面提取）
    retriever/
      hybrid_retriever.py  混合检索（BM25 + 向量 RRF 融合 + 重排）
      bm25_retriever.py    BM25 关键词检索（jieba 分词）
      query_rewriter.py    LLM 查询改写
      reranker.py          LLM / CrossEncoder 重排
  security/
    input_filter.py        输入安全过滤
  storage/
    book_catalog.py        SQLite 目录（BookCatalog / NoteCatalog / PlanCatalog）
    note_storage.py        笔记文件 I/O（LocalNoteStorage）
    plan_storage.py        阅读计划文件 I/O（LocalPlanStorage）
  xai/
    citation.py            Citation 引用构建

data/                      运行时数据（不提交 Git）
  books/                   EPUB 书库
  books.db                 书籍/笔记/计划元数据（SQLite）
  chroma/                  ChromaDB 向量库（本地模式）
  covers/                  EPUB 封面图片
  notes/                   笔记 Markdown 文件（以 book_id 命名）
  plans/                   阅读计划 Markdown 文件（以 book_id 命名）
  checkpoints.db           LangGraph 多轮对话检查点（SQLite）

scripts/
  rag_demo.py              端到端 RAG 演示
  deepread_graph_demo.py   LangGraph Agent 演示

tests/
  rag/                     RAG 组件单元测试
  agents/                  Agent 单元测试
  api/                     API 端点单元测试
  storage/                 Storage 单元测试
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 `.env`

```env
OPENAI_API_KEY=sk-xxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

CHROMA_PERSIST_DIR=data/chroma
BOOKS_DATA_DIR=data/books

# Chroma Cloud（留空则使用本地模式）
CHROMA_API_KEY=
CHROMA_TENANT=default_tenant
CHROMA_DATABASE=default_database

# Mem0 长期记忆用户 ID
MEM0_USER_ID=default_user

# 笔记和阅读计划存储目录
NOTE_STORAGE_DIR=data/notes
PLAN_STORAGE_DIR=data/plans

# SQLite 目录数据库（书籍/笔记/计划元数据）
BOOK_CATALOG_DB=data/books.db

# EPUB 封面提取目录
COVERS_DIR=data/covers
```

### 3. 启动 API

```bash
uvicorn backend.main:app --reload
```

### 4. 上传 EPUB 书籍

```bash
curl -X POST http://localhost:8000/books/upload \
  -F "file=@path/to/your/book.epub"
```

响应：

```json
{
  "book_id": "a1b2c3d4-...",
  "source": "data/books/book.epub",
  "collection_name": "kant_library",
  "total_chunks": 860,
  "added": 860,
  "skipped": 0
}
```

### 5. 对话

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "康德的先验统觉是什么？", "thread_id": "session-1"}'
```

响应：

```json
{
  "answer": "精读回答：...",
  "citations": [{"book_title": "...", "section": "..."}],
  "retrieved_docs_count": 6,
  "intent": "deepread"
}
```

---

## API 参考

### POST /chat

**请求字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `query` | string | 用户输入 |
| `book_id` | string? | 指定书的 UUID（限定检索范围） |
| `thread_id` | string | 会话 ID，同一 ID 共享多轮上下文（默认 `"default"`） |
| `active_tab` | string? | 前端当前标签页，直接路由到对应 Agent，跳过意图分类 |
| `selected_text` | string? | 用户在阅读器中划选的原文，注入为问题上下文 |
| `current_chapter` | string? | 当前阅读章节，注入为问题上下文 |

**响应字段：**

| 字段 | 说明 |
|------|------|
| `answer` | Agent 回答（Markdown） |
| `citations` | 引用列表（`book_title`, `section_index` 等） |
| `retrieved_docs_count` | 检索到的文档数量 |
| `intent` | 识别到的意图（`deepread` / `notes` / `plan` / `recommend`） |

### GET /books

- 返回所有书籍列表，来自 SQLite `BookCatalog`
- 每条包含 `book_id`、`title`、`author`、`status`、`progress`、`cover_path` 等完整字段

### POST /books/upload

- 上传 `.epub` 文件（multipart/form-data）
- 触发入库流水线：解析 → 切块 → 向量化 → 写入 ChromaDB → 写入 SQLite → 提取封面
- 入库成功后自动刷新 BM25 索引缓存
- 响应含 `book_id`（确定性 uuid5，后续所有操作均以此为主键）

### POST /reader/{book_id}/init

- 用户打开书籍时调用，幂等
- 自动生成阅读计划（提取全部章节 + LLM 建议日程），写入 `data/plans/{book_id}.md`
- 更新书籍状态为 `"reading"`

### GET /reader/{book_id}/plan

- 返回当前阅读计划 Markdown，通过 `PlanCatalog` 定位文件

### POST /reader/{book_id}/progress

- 请求体：`{"chapter": "章节名"}`
- 将对应章节标记为已读（`- [ ]` → `- [x]`），重新计算完成进度，同步到 `BookCatalog`

---

## 核心功能说明

### 五大核心 Agent

为了让你更容易理解，我们把这个系统想象成一个“阅读工作小组”，每个人（Agent）有自己明确的分工：

#### 1. RouterAgent（路由主管）
**它的工作**：就像医院的导诊台。
当你发出一句话时，它会先判断你是在“闲聊”还是在“认真提问”。如果是认真提问，它会帮你把问题“翻译”得更专业，方便后面去书里找答案。

#### 2. DeepReadAgent（精读专员）
**它的工作**：核心答疑老师。
它会去书里翻找最相关的段落，然后严格按照书里的原文来回答你的问题。为了证明它没有胡说八道，它的每一句话都会附带上引用的原文出处。

#### 3. CriticAgent（事实核查评论员）
**它的工作**：专门挑刺的质检员。
在精读专员给出回答后，它会悄悄在旁边检查：“这个回答客观吗？是不是有偏见？书里真的这么说了吗？”如果发现问题，它会给你发一个小小的“审查笔记”来补充客观视角，防止 AI 产生幻觉。

#### 4. NoteAgent（笔记小助手）
**它的工作**：自动整理笔记。
你不需要每次问完问题都手动记笔记。每次问答结束后，它会自动从你们的对话里提炼出“核心要点”和“关键概念”，并帮你整整齐齐地写进这本专属的笔记文档里。

#### 5. FollowupAgent（引导思考员）
**它的工作**：启发式提问。
回答完你的问题后，它会根据刚才聊的内容，顺势给你提出 3 个有启发性的小问题，引导你进一步深入思考。

### 多 Agent 协作流水线

简单来说，你只问了一句话，但系统后台其实有一群人在协同工作：

```text
你问："康德的先验统觉是什么？"

流水线开始：
1. 【路由主管】 识别出这是个专业问题，优化了查询词。
2. 【精读专员】 拿着词去书里找答案，并生成了带引用的解释。
3. 【质检员】   (同步进行) 检查这个解释是否客观、严谨。
4. 【笔记助手】 (同步进行) 把这轮精彩问答浓缩成笔记存了起来。
5. 【引导员】   (同步进行) 顺势抛出 3 个更深入的问题供你继续探讨。
```

### 多轮对话

同一 `thread_id` 下的对话历史保存在 `data/checkpoints.db`（SQLite），对话可以跨请求延续：

```
Turn 1: "康德的先验统觉是什么？"  → deepread
Turn 2: "它和范畴的关系呢？"       → deepread（记得上轮内容）
Turn 3: "推荐几本认识论相关的书"   → recommend（切换 Agent，上下文保留）
Turn 4: "帮我制定4周阅读计划"      → plan（继续同一会话）
```

### 长期记忆

Mem0 将每轮对话提炼为语义摘要，跨会话保留用户画像（阅读偏好、已读书目、知识背景等），在新会话开始时自动注入，让每次对话都"认识"用户。

### 混合检索（HybridRetriever）

```
用户问题
  ├── QueryRewriter（LLM 改写为更适合检索的形式）
  ├── 向量检索（语义相似，top-20）
  └── BM25 关键词检索（精确匹配，top-20）
        ↓
     RRF 融合（互惠排名融合）
        ↓
     LLM Reranker（从候选中挑最相关的 6 篇）
```

BM25 索引按书源（filter）分开缓存，上传新书后自动刷新。

---

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 带覆盖率
pytest tests/ --cov=backend --cov-report=html

# 单模块
pytest tests/agents/test_recommendation_agent.py -v
```

所有单元测试使用内存 fixture 和 mock，无需真实 EPUB 文件或 API 调用。

---

## 安全

- `.env` 已在 `.gitignore` 中忽略，提交项目时只提供 `.env.example`
- 每次请求都经过 `InputSafetyFilter` 检查，拦截 API Key 泄露、提示词注入等
- Chroma Cloud 模式下 API Key 通过环境变量传入，不写入代码
