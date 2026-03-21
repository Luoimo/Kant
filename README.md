# Kant — 读书 AI 助手

基于 **OpenAI + ChromaDB + LangGraph** 构建的本地读书 Agent 系统，支持 EPUB 书籍的全流程 RAG 问答、笔记整理、推荐和阅读计划。

---

## 架构概览

```
EPUB → EpubExtractor → TextCleaner → TextChunker → OpenAI Embeddings → ChromaDB
                                                                           ↓
用户问题 → FastAPI /chat → OrchestratorAgent (LangGraph)
                               ├─ Mem0 记忆检索
                               ├─ 安全过滤 (InputSafetyFilter)
                               ├─ 意图识别 (LLM 结构化输出，支持 compound_intents)
                               └─ 路由到子 Agent（可多 Agent 串联）：
                                   ├─ DeepReadAgent      (精读 / 问答)
                                   ├─ NoteAgent          (笔记整理，含持久化)
                                   ├─ ReadingPlanAgent   (阅读计划，含持久化)
                                   └─ RecommendationAgent (书籍推荐)
```

---

## 目录结构

```
backend/
  agents/          LangGraph Agent 系统（orchestrator + 4 个子 Agent）
  api/chat.py      FastAPI 入口（POST /chat）
  llm/             OpenAI LLM & Embeddings 封装
  memory/          Mem0 长期记忆
  rag/
    chroma/        ChromaStore（向量库管理）
    chunker/       文本切块
    cleaner/       文本清洗
    extracter/     EPUB 提取
    retriever/     混合检索（BM25 + 向量 + 重排）
  security/        InputSafetyFilter 安全过滤
  storage/         笔记 / 阅读计划持久化（NoteStorage / PlanStorage）
  xai/             Citation 引用构建
data/
  chroma/          ChromaDB 向量库（本地模式）
  books/           EPUB 书库
  notes/           笔记 Markdown 文件（LocalNoteStorage 默认目录）
  plans/           阅读计划 Markdown 文件（LocalPlanStorage 默认目录）
  checkpoints.db   LangGraph SqliteSaver 会话检查点
scripts/
  rag_demo.py      端到端 RAG Demo
  deepread_graph_demo.py  LangGraph Agent 演示
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

# Mem0 记忆用户 ID
MEM0_USER_ID=default_user

# 笔记 / 阅读计划持久化目录
NOTE_STORAGE_DIR=data/notes
PLAN_STORAGE_DIR=data/plans
```

### 3. 入库 EPUB

将 `.epub` 文件放入 `data/books/`，然后在 `scripts/rag_demo.py` 中取消注释 `ingest_books`，运行：

```bash
python -m scripts.rag_demo
```

### 4. 启动 API

```bash
uvicorn backend.main:app --reload
```

POST `http://localhost:8000/chat`：

```json
{
  "query": "克尔凯郭尔如何定义焦虑？",
  "book_source": null,
  "thread_id": "session-1"
}
```

---

## 核心模块说明

### RAG 流水线

`ChromaStore` 是唯一入口，负责编排：

1. **EpubExtractor** — 提取 EPUB 章节文本与元数据
2. **TextCleaner** — 去除页眉页脚、噪声块
3. **TextChunker** — 按 `chunk_size` / `chunk_overlap` 切块，带 `chunk_id`（内容哈希）去重
4. **OpenAI Embeddings** — 批量向量化，写入 ChromaDB

检索方法：
- `similarity_search(query, k, filter)` — 纯向量检索
- `as_retriever()` — LangChain 兼容 Retriever

`backend/rag/retriever/` 提供混合检索组件：BM25 + 向量 RRF 融合、QueryRewriter（查询改写）、LLMReranker / CrossEncoderReranker（重排序）。**所有四个子 Agent 均使用 `HybridRetriever`**，在构造时一次性初始化。

### Agent 系统

基于 LangGraph Supervisor 模式，图结构：

```
START → memory_search → supervisor → [agent(s)] → supervisor → finalize → END
```

- **memory_search** — 从 Mem0 检索历史记忆，注入 `memory_context`
- **supervisor** — 安全过滤 → 意图识别（支持 `compound_intents` 多 Agent 串联）→ 路由
- **DeepReadAgent** — 精读问答，HybridRetriever 检索 top-k 文档，构建 `Citation`，可选一致性自检
- **NoteAgent** — 4 种格式（structured / summary / qa / timeline），支持 new / edit / extend，笔记持久化到 `data/notes/`
- **RecommendationAgent** — hash 缓存书目，多轮去重推荐，带评级和阅读建议
- **ReadingPlanAgent** — 从 ChromaDB 提取真实章节结构，计算阅读时长，支持进度更新，计划持久化到 `data/plans/`
- **finalize** — 多 Agent 串联时合成统一回答；将本轮问答存入 Mem0 长期记忆

**多轮对话：** `thread_id` 隔离会话，`SqliteSaver`（`data/checkpoints.db`）持久化检查点，每个 Agent 维护独立的消息历史（`*_messages`）。

### 持久化存储

`backend/storage/` 提供可插拔的文件存储：

- `LocalNoteStorage` — 笔记 Markdown 文件，存储于 `NOTE_STORAGE_DIR`（默认 `data/notes/`）
- `LocalPlanStorage` — 阅读计划 Markdown 文件，存储于 `PLAN_STORAGE_DIR`（默认 `data/plans/`）
- 均实现 `Protocol`（runtime-checkable），可替换为自定义后端
- 操作：`save` / `load` / `list` / `delete`，文件名使用 `note_id` / `plan_id` + `.md` 后缀

### Chroma 双模式

- `CHROMA_API_KEY` 为空 → `PersistentClient`（本地落盘）
- `CHROMA_API_KEY` 非空 → `CloudClient`（Chroma Cloud）

### 安全层

`InputSafetyFilter` 硬阻断：API Key 泄露、文件系统命令、提示词注入、代码执行关键词。软警告：非阅读相关查询。

---

## 测试

```bash
pytest tests/ -v
pytest tests/rag/test_chroma_store.py -v
pytest tests/ --cov=backend --cov-report=html
```

所有单元测试使用内存 fixture，无需真实 PDF/EPUB 或 API 调用。

---

## 安全

`.env` 已在 `.gitignore` 中忽略。分享项目时只提供 `.env.example`，不要提交真实凭据。
