# Kant — 读书 AI 助手

基于 **OpenAI + ChromaDB + LangGraph** 构建的本地读书 Agent 系统，支持 EPUB 书籍的精读问答、笔记整理、书单推荐和阅读计划，具备多轮对话与长期记忆能力。

---

## 架构概览

```
EPUB → EpubExtractor → TextCleaner → TextChunker → OpenAI Embeddings → ChromaDB
                                                                           ↓
用户问题 → FastAPI /chat → OrchestratorAgent (LangGraph)
                               ├─ Mem0 长期记忆检索
                               ├─ InputSafetyFilter（每轮安全检查）
                               ├─ 意图识别（LLM 结构化输出，支持多 Agent 串联）
                               └─ 路由到子 Agent：
                                   ├─ DeepReadAgent      精读问答（证据引用）
                                   ├─ NoteAgent          笔记整理（持久化）
                                   ├─ ReadingPlanAgent   阅读计划（持久化）
                                   └─ RecommendationAgent 书籍推荐（LLM 知识）
```

---

## 目录结构

```
backend/
  agents/
    orchestrator_agent.py  LangGraph 图 + Supervisor 逻辑
    deepread_agent.py      精读问答 Agent
    note_agent.py          笔记 Agent
    reading_plan_agent.py  阅读计划 Agent
    recommendation_agent.py 书籍推荐 Agent
  api/
    chat.py                FastAPI 路由（POST /chat, POST /books/upload）
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
    extracter/             EPUB 解析（EpubExtractor）
    retriever/
      hybrid_retriever.py  混合检索（BM25 + 向量 RRF 融合 + 重排）
      bm25_retriever.py    BM25 关键词检索（jieba 分词）
      query_rewriter.py    LLM 查询改写
      reranker.py          LLM / CrossEncoder 重排
  security/
    input_filter.py        输入安全过滤
  storage/
    note_storage.py        笔记持久化（LocalNoteStorage）
    plan_storage.py        阅读计划持久化（LocalPlanStorage）
  xai/
    citation.py            Citation 引用构建

data/                      运行时数据（不提交 Git）
  books/                   EPUB 书库
  chroma/                  ChromaDB 向量库（本地模式）
  notes/                   笔记 Markdown 文件
  plans/                   阅读计划 Markdown 文件
  checkpoints.db           LangGraph 多轮对话检查点（SQLite）

scripts/
  rag_demo.py              端到端 RAG 演示
  deepread_graph_demo.py   LangGraph Agent 演示

tests/
  rag/                     RAG 组件单元测试
  agents/                  Agent 单元测试
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
| `book_source` | string? | 指定书的文件路径（限定检索范围） |
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

### POST /books/upload

- 上传 `.epub` 文件（multipart/form-data）
- 触发入库流水线：解析 → 切块 → 向量化 → 写入 ChromaDB
- 入库成功后自动刷新 BM25 索引缓存

---

## 核心功能说明

### 四种 Agent

#### DeepReadAgent — 精读问答
从书库中检索最相关的段落，严格基于原文回答，每条回答附带章节引用。使用混合检索（BM25 + 向量 + RRF 融合 + LLM 重排）确保检索质量。

#### NoteAgent — 笔记整理
支持 4 种笔记格式：`structured`（结构化）/ `summary`（摘要）/ `qa`（问答）/ `timeline`（时间线）。支持 `new`（新建）/ `edit`（编辑）/ `extend`（追加）操作，笔记以 Markdown 文件存储在 `data/notes/`。

#### RecommendationAgent — 书籍推荐
基于大模型自身知识推荐书籍，不局限于本地书库，可以推荐用户尚未上传的书。从本地书库获取已有书目仅用于标注（`✅ 已在库` / `📥 可上传精读`）。多轮对话中自动避免重复推荐。

#### ReadingPlanAgent — 阅读计划
从 ChromaDB 中提取真实章节结构，按阅读速度（300字/分钟）计算每章时长，生成个性化进度安排。支持 `new` / `edit` / `extend` 操作，计划以 Markdown 文件存储在 `data/plans/`。若指定的书不在书库中，基于大模型知识生成计划并注明来源限制。

### 多 Agent 串联

一句话可以触发多个 Agent 依次工作：

```
"推荐一本海德格尔的书，并帮我制定阅读计划"
  → RecommendationAgent 推荐《存在与时间》
  → ReadingPlanAgent 为《存在与时间》生成阅读计划
  → finalize 节点合并两个结果为一份完整回答
```

```
"帮我分析康德范畴演绎，然后整理成笔记"
  → DeepReadAgent 精读检索
  → NoteAgent 基于精读上下文生成笔记
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

所有单元测试使用内存 fixture，无需真实 EPUB 文件或 API 调用（185 个测试）。

---

## 安全

- `.env` 已在 `.gitignore` 中忽略，提交项目时只提供 `.env.example`
- 每次请求都经过 `InputSafetyFilter` 检查，拦截 API Key 泄露、提示词注入等
- Chroma Cloud 模式下 API Key 通过环境变量传入，不写入代码
