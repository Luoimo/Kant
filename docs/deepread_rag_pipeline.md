# DeepReadAgent RAG 检索流程

## 整体流程图

```
用户问题 (query)
    │
    ▼
┌─────────────────────────────────────────────────┐
│ QueryRewriter（LLM 改写，可选）                   │
│  · 展开代词，补充哲学专业术语（中/德/拉丁文对应词）  │
│  · 失败时自动降级，使用原始 query                  │
└──────────────────┬──────────────────────────────┘
                   │  rewritten_query
       ┌───────────┴───────────┐
       ▼                       ▼
┌─────────────┐       ┌──────────────────────┐
│ 向量检索     │       │ BM25 关键词检索        │
│ (ChromaDB)  │       │ (BM25Okapi + jieba)  │
│ fetch_k=20  │       │ fetch_k=20           │
│ 余弦相似度   │       │ 词频-逆文档频率        │
└──────┬──────┘       └──────────┬───────────┘
       │  [(Doc, dist)]          │  [(Doc, score)]
       └───────────┬─────────────┘
                   ▼
        ┌──────────────────┐
        │ 去重（chunk_id）  │
        │ RRF 融合          │
        │ score = Σ 1/(k+rank) │
        │ rrf_k = 60       │
        └────────┬─────────┘
                 │  候选文档（按 RRF 分数排序）
                 ▼
        ┌──────────────────┐
        │ LLMReranker      │
        │ 对每个候选段落打分 │
        │ (0-10 整数)      │
        │ 取 top final_k=6 │
        └────────┬─────────┘
                 │  final_docs (≤6 个)
                 ▼
        ┌─────────────────────────────┐
        │ 构建 Citation               │
        │ 从 metadata 提取：          │
        │  · book_title               │
        │  · chapter_title            │
        │  · section_title            │
        │  · section_indices          │
        └────────┬────────────────────┘
                 │
                 ▼
        ┌──────────────────────────────────┐
        │ LLM 生成回答（DeepRead Prompt）  │
        │  · 最多使用 max_evidence=8 篇证据 │
        │  · 每条结论标注书名+章节          │
        │  · 证据不足时明确说明            │
        │  · 注入 memory_context（可选）   │
        └────────┬─────────────────────────┘
                 │
                 ▼
        ┌──────────────────────────┐
        │ 一致性自检（可选）         │
        │ consistency_check=False  │
        │ 第二次 LLM 调用审查回答   │
        └──────────────────────────┘
                 │
                 ▼
         DeepReadResult
          ├── answer: str
          ├── citations: list[Citation]
          ├── retrieved_docs: list[Document]
          ├── consistency_ok: bool | None
          └── consistency_feedback: str | None
```

---

## 各阶段详解

### 1. QueryRewriter — 查询改写

**文件：** `backend/rag/retriever/query_rewriter.py`

LLM 接收用户原始问题，输出更适合检索的改写版本。

改写策略（系统 prompt 规定）：
- 展开代词，使问题自给自足
- 补充中文/德文/拉丁文哲学专业术语
- 保留原始意图，不过度扩展

失败处理：LLM 异常时直接使用原始 query，不中断流程。

配置：`HybridConfig.enable_query_rewrite = True`（默认开启）

---

### 2. 双路检索

#### 向量检索（Dense）

**文件：** `backend/rag/chroma/chroma_store.py` → `similarity_search_with_score`

- 用 OpenAI Embeddings 将改写后的 query 向量化
- 在 ChromaDB 中做余弦相似度查询
- 返回 `fetch_k`（默认 20）个 `(Document, distance)` 对
- 支持 `filter={"source": "book.epub"}` 限定书籍范围

#### BM25 关键词检索（Sparse）

**文件：** `backend/rag/retriever/bm25_retriever.py`

- 首次调用时从 ChromaDB 拉取全量文档（`get_all_documents`）构建倒排索引
- 后续查询复用同一索引（**懒加载，进程内缓存**）
- 使用 `jieba` 中文分词；未安装时退回字符级 tokenize
- 空语料库时安全返回空列表（不崩溃）

---

### 3. RRF 融合（Reciprocal Rank Fusion）

**文件：** `backend/rag/retriever/hybrid_retriever.py` → `_rrf_fusion`

```
score(doc) = Σ  1 / (rrf_k + rank)
```

- `rrf_k = 60`（平滑常数，抑制高位排名的过度优势）
- 向量和 BM25 各贡献一个排序列表
- 先按 `chunk_id` 去重，再融合两路分数
- 融合后按 RRF 分数降序排列，得到候选池

**优势：** 对专有名词（BM25 擅长）和语义相关（向量擅长）都有良好召回，互补效果明显。

---

### 4. LLMReranker — 精排

**文件：** `backend/rag/retriever/reranker.py`

将候选文档（RRF 排序后）送入 LLM，要求对每个段落打 0-10 分，取 `final_k`（默认 6）个最高分文档。

```
prompt: 问题 + 候选段落（每段截取前500字）
输出:   1: 8
        2: 3
        ...
```

失败处理：LLM 异常时按 RRF 顺序直接截断，不中断流程。

替代方案：`CrossEncoderReranker`（需 `pip install sentence-transformers`），使用本地 `BAAI/bge-reranker-base` 模型，无需额外 API 调用。

---

### 5. 证据组装与回答生成

**文件：** `backend/agents/deepread_agent.py` → `_answer_with_evidence`

- 取 `max_evidence`（默认 8）篇文档拼入 prompt
- 每篇标注：书名、章节名
- 系统 prompt 硬性规则：
  - 只引用证据中明确出现的内容
  - 对关键结论标注来源（书名·章节）
  - 证据不足时明确说明，不编造

---

### 6. 一致性自检（可选）

`DeepReadConfig.consistency_check = False`（默认关闭）

开启后对生成的回答做第二次 LLM 审查，判断结论是否有证据支撑，输出 `consistency_ok` 和 `consistency_feedback`。

---

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `k` | 6 | rerank 后最终保留的文档数 |
| `fetch_k` | 20 | 向量/BM25 各自的候选召回数 |
| `max_evidence` | 8 | 拼入 prompt 的证据上限 |
| `rrf_k` | 60 | RRF 平滑常数 |
| `reranker` | `"llm"` | `"llm"` / `"cross_encoder"` / `"none"` |
| `enable_query_rewrite` | `True` | 是否启用 LLM 查询改写 |
| `consistency_check` | `False` | 是否开启一致性自检（额外 LLM 调用） |

通过 `DeepReadConfig` 传入 `DeepReadAgent`：

```python
from backend.agents.deepread_agent import DeepReadAgent, DeepReadConfig
from backend.rag.retriever import HybridConfig

agent = DeepReadAgent(
    store=store,
    config=DeepReadConfig(
        k=6,
        fetch_k=20,
        consistency_check=True,
        hybrid=HybridConfig(
            fetch_k=20,
            final_k=6,
            reranker="llm",
            enable_query_rewrite=True,
        ),
    ),
)
```

---

## 关键文件索引

| 职责 | 文件 |
|------|------|
| Agent 入口 & 回答生成 | `backend/agents/deepread_agent.py` |
| 混合检索编排 | `backend/rag/retriever/hybrid_retriever.py` |
| 查询改写 | `backend/rag/retriever/query_rewriter.py` |
| BM25 关键词检索 | `backend/rag/retriever/bm25_retriever.py` |
| LLM / CrossEncoder 重排 | `backend/rag/retriever/reranker.py` |
| 向量库（ChromaDB 封装） | `backend/rag/chroma/chroma_store.py` |
| Citation 构建 | `backend/xai/citation.py` |
| 检索器单元测试 | `tests/rag/test_retriever.py` |
