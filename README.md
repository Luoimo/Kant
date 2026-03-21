## 项目说明（Kant RAG Demo）

本项目是一个基于 **OpenAI + Chroma** 的本地 RAG（Retrieval-Augmented Generation）示例，支持：

- **本地 Chroma 向量库**（默认，落盘到 `data/chroma`）
- **Chroma Cloud 模式**（使用官方托管向量库，通过 API Key 访问）

下面说明如何配置 `.env`，以及当前 RAG 的整体流程与运行方式。

---

## 一、环境与依赖

- Python 3.10+（建议）
- 已创建虚拟环境并安装依赖：

```bash
pip install -r requirements.txt
```

项目根目录结构（节选）：

- `backend/config.py`：读取 `.env` 的配置中心
- `backend/llm/openai_client.py`：OpenAI LLM & Embeddings 封装
- `backend/rag/chroma/chroma_store.py`：向量库管理（Chroma 包装）
- `backend/rag/extracter/`：PDF 文本抽取
- `backend/rag/cleaner/`：文本清洗
- `backend/rag/chunker/`：文本切块
- `scripts/rag_demo.py`：端到端 RAG Demo 脚本

---

## 二、`.env` 配置说明

根目录已有一个 `.env` 模板，关键字段如下（**请根据自己环境修改**）：

```env
# OpenAI（必填）
OPENAI_API_KEY=你的_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# 本地向量库目录
CHROMA_PERSIST_DIR=data/chroma
BOOKS_DATA_DIR=data/books

# Chroma Cloud（可选：填了就走 CloudClient，留空则使用本地 PersistentClient）
CHROMA_API_KEY=你的_chroma_api_key
CHROMA_TENANT=你的_tenant_id
CHROMA_DATABASE=你的_database_name
```

### 1. OpenAI 配置

- `OPENAI_API_KEY`：从 OpenAI 控制台获取的 API Key。
- `OPENAI_BASE_URL`：如使用代理或自建网关，可改为对应地址。
- `OPENAI_MODEL`：对话 / 推理模型名称。
- `OPENAI_EMBEDDING_MODEL`：Embedding 模型名称，用于向量化文本。

> 所有这些字段由 `backend/llm/openai_client.py` 统一读取。

### 3. Chroma Cloud 模式

如需使用 **Chroma Cloud**：

1. 在 Chroma Cloud 后台创建：
   - 一个 Tenant（tenant id）
   - 一个 Database（database name）
   - 一个 API Key
2. 在 `.env` 中填入：

```env
CHROMA_API_KEY=ck_xxx           # 必填：Cloud API Key
CHROMA_TENANT=xxxx-tenant-id    # 必填：Tenant ID
CHROMA_DATABASE=Kant            # 必填：Database 名称
```

此时 `backend/rag/chroma/chroma_store.py` 内部会自动选择：

- `chromadb.CloudClient(tenant=..., database=..., api_key=...)`

而 `collection_name` 默认设置为 `CHROMA_DATABASE`，便于 Cloud / 本地统一管理。

---

## 三、当前 RAG 流程概览

核心流程由 `ChromaStore` 负责（`backend/rag/chroma/chroma_store.py`）：

1. **PDF 提取（Extract）**
   - 使用 `PDFExtractor` 读取 `BOOKS_DATA_DIR` 中的 PDF。
   - 输出 `PDFContent`，包含每页的文本与元信息（标题、作者等）。

2. **文本清洗（Clean）**
   - `TextCleaner` 根据 `CleanConfig`：去除页眉页脚、图片块、过短块等噪声。
   - 输出结构化的 `CleanedContent`。

3. **文本切块（Chunk）**
   - `TextChunker` 根据 `ChunkConfig`（如 `chunk_size`、`chunk_overlap`）进行分段。
   - 每个 Chunk 对应一个 `TextChunk`，带有：
     - `chunk_id`（基于内容的哈希）
     - `source`（PDF 路径）
     - `page_numbers`（涉及页码）
     - `pdf_title` / `pdf_author` 等元数据。

4. **向量化 & 入库（Embed & Ingest）**
   - 使用 `get_embeddings()` 调用 OpenAI Embeddings，将 Chunk 文本向量化。
   - 调用 `ChromaStore._ingest_chunks_to_db()`：
     - 支持 **按 `chunk_id` 去重**（`skip_existing=True`）；
     - 支持 **分批写入**（`embed_batch_size` 控制）。
   - 元数据写入 Chroma / Chroma Cloud，对应 collection 中的 metadatas。

5. **检索（Retrieve）**
   - `similarity_search(query, k, filter)`：返回最相似的 k 个 `Document`。
   - `similarity_search_with_score`：同时返回距离分数。
   - `as_retriever(...)`：返回 LangChain 兼容的 Retriever，可直接接入 Chain / Agent。

> 目前 Demo 仅做检索和结果打印；你可以在此基础上增加一个 LLM，将检索结果拼接成 RAG 回答。

---

## 四、运行 RAG Demo

Demo 脚本：`scripts/rag_demo.py`

### 1. 准备数据

将你的 PDF 放到：

```text
data/books/
```

如目录不存在，可以手动创建，或在代码中修改 `BOOKS_DATA_DIR`。

### 2. 首次入库（Ingest）

在 `scripts/rag_demo.py` 中，默认 `ingest_books` 是注释掉的：

```python
def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    books_dir = project_root / "data" / "books"

    store = build_store()

    # 首次运行时取消注释以写入数据；后续查询无需重复写入（skip_existing=True）
    # ingest_books(store, books_dir)

    run_query(store, "克尔凯郭尔对焦虑的定义是什么？", k=5)
    run_query(store, "克尔凯郭尔的父亲是个怎样的人？", k=5)
```

**第一次运行时**：

1. 取消注释 `ingest_books(store, books_dir)`。
2. 执行：

```bash
python -m scripts.rag_demo
```

写入完成后，可以再把 `ingest_books` 注释回去，避免重复入库。

### 3. 检索测试（Query）

只保留两行 `run_query`，再次执行：

```bash
python -m scripts.rag_demo
```

终端会打印若干命中结果，包括：

- 来源文件 `source`
- 页码 `page_numbers`
- `chunk_index`
- 书名 `pdf_title`
- 作者 `pdf_author`
- 截断后的内容片段

---

## 五、安全注意事项

- `.env` 中包含 **OpenAI / Chroma API Key**，已经在 `.gitignore` 中忽略，**不要手动提交**。
- 如果要分享项目，请仅分享代码和示例配置（可以提供 `.env.example`），不要分享真实凭据。

