### RAG 模块总体概览

当前 `backend/rag` 模块实现了一条完整、可测试的 PDF→文本→清洗→切块→向量库（Chroma）流水线，并且对配置与元数据做了比较细致的封装，便于后续扩展到多文档、多集合场景。

整体数据流如下：

```text
PDF 文件
  └─ PDFExtractor（提取）
       └─ PDFContent / PageContent
            └─ TextCleaner（清洗）
                 └─ CleanedContent / CleanedPage
                      └─ TextChunker（切块）
                           └─ TextChunk 列表
                                └─ ChromaStore（向量化 + 存储 + 检索）
                                     └─ Chroma / OpenAI Embeddings
```

---

## 1. 文本提取：`extracter/pdf_extractor.py`

**核心类：**

- `PDFExtractor`
  - 基于 PyMuPDF (`fitz`) 对 PDF 进行解析。
  - 支持按页范围提取 / 全文提取 / 逐页迭代 / 元数据读取。
- 数据结构：
  - `PageContent`
    - `page_number`：页码（1-based）
    - `text`：整页纯文本
    - `blocks`：来自 `page.get_text("blocks")` 的块信息（`bbox/text/block_no/block_type`）
    - `images`：图片信息（`xref` + `bbox`，可选）
    - `width/height`：页面尺寸
  - `PDFContent`
    - `source`：源文件路径
    - `total_pages`：总页数
    - `metadata`：PDF 元数据（title/author/subject/keywords/...）
    - `pages: list[PageContent]`
    - 属性 `full_text`：按页拼接后的整本文本（以 `\n` 作为分隔）

**主要方法：**

- `__init__(path: str | Path)`
  - 校验路径存在、后缀为 `.pdf`，否则抛 `FileNotFoundError` / `ValueError`。
- `extract(start_page=1, end_page=None, extract_images=False) -> PDFContent`
  - 提取指定页区域的文本、块、（可选）图片信息。
  - `total_pages` 始终为整本 PDF 页数，即使只取了部分页面。
- `iter_pages(start_page=1, end_page=None, extract_images=False) -> Iterator[PageContent]`
  - 懒迭代，适合大文件流式处理。
- `extract_text_only(start_page=1, end_page=None) -> str`
  - 仅返回纯文本（`page.get_text("text")` 的拼接），不构造块结构。
- `get_metadata() -> dict`
  - 直接读取 PDF 元数据，无需遍历页面。
- `page_count -> int`
  - 快速获取总页数。

---

## 2. 文本清洗：`cleaner/text_cleaner.py`

**配置类：**

- `CleanConfig`
  - 对清洗行为进行细粒度开关控制，默认开启一套安全、通用的策略：
  - 来自 `unstructured.cleaners` 的选项：
    - `extra_whitespace`：压缩多余空白（空格、换行等）
    - `bullets`：去掉无序列表符号（•、◦ 等）
    - `dashes`：统一破折号风格
    - `trailing_punctuation`：去掉行尾多余标点
    - `ordered_bullets`：移除有序编号（`1. `、`a)` 等）
    - `unicode_quotes`：统一花引号（在代码中做了额外兜底）
  - 自定义选项：
    - `fix_hyphenation`：修复跨行连字符断词（`word-\nword` → `wordword`）
    - `remove_page_numbers`：移除孤立页码行（`42`、`第42页`、`xiv` 等）
    - `remove_headers_footers`：基于 `bbox` + 页高启发式删页眉页脚
    - `header_footer_lines`：页眉/脚阈值（以“行高”约 20px 计）
    - `normalize_unicode`：NFC 规范化 Unicode
    - `min_block_chars`：文本块最小长度，小于则视为噪声

**数据结构：**

- `CleanedPage`
  - 对应 `PageContent` 的清洗版：
  - `page_number/text/blocks/width/height/source_page`
- `CleanedContent`
  - 整个 PDF 的清洗结果：
  - `source/total_pages/metadata/pages: list[CleanedPage]`
  - 属性 `full_text`：拼接非空页文本，页间使用双换行 `\n\n`

**核心类：**

- `TextCleaner`

**主要方法：**

- `clean_text(text: str) -> str`
  - 纯字符串管道：Unicode 规范化 → 修复连字符断词 → 删除页码行 → Unicode 引号替换 → `unstructured.clean` → 合并损坏的段落 → 最终压缩空白。
  - 对制表符 `\t` 专门替换为空格，避免残留。
- `clean_page(page: PageContent) -> CleanedPage`
  - 过滤块（图片块、短块、页眉/页脚、孤立页码）→ 重建页文本 → 走 `clean_text`。
- `clean_pages(pages: Sequence[PageContent]) -> list[CleanedPage]`
- `clean_content(content: PDFContent) -> CleanedContent`
  - 全书级清洗，保留源文件路径、页数和元数据（做浅拷贝，防止副作用）。

内部辅助：

- `_is_page_number(line: str) -> bool`
  - 正则识别“孤立页码”（阿拉伯、罗马数字、`第X页` 等）。
- `_is_header_footer(blk: dict, page_height: float) -> bool`
  - 根据 `bbox.y0/y1` 距页面上下边缘的距离和 `header_footer_lines` 判断是否为页眉/脚。

---

## 3. 文本切块：`chunker/text_chunker.py`

**配置类：**

- `ChunkConfig`
  - `splitter`: `"recursive"` / `"token"`
  - `chunk_size`: 递归拆分时为最大字符数，token 拆分时为最大 token 数
  - `chunk_overlap`: 相邻块的重叠量（字符或 token）
  - `encoding_name`: token 模式的 tiktoken 编码（默认 `cl100k_base`）
  - `separators`: 递归拆分分隔符优先级列表（`"\n\n" → "\n" → "。"` 等）
  - `page_aware`: 
    - `True`：逐页切分，每个块关联单一页码；
    - `False`：全文拼接后统一切分，块可跨多页。
  - `min_chunk_chars`: 块最小长度，短块过滤（避免向量库塞入噪声）。

**数据结构：**

- `ChunkMeta`
  - `source`: 源文件路径
  - `page_numbers`: 该块涉及的页码列表（1-based）
  - `chunk_index`: 文档级全局顺序
  - `pdf_title/pdf_author`: 元信息
- `TextChunk`
  - `chunk_id`: 内容的 SHA-256 前 16 位 hex（用于去重）
  - `text`: 块文本
  - `char_count`: 字符数
  - `metadata: ChunkMeta`
  - 方法 `to_dict()`：扁平字典表示，便于写入其他系统或落盘。

**核心类：**

- `TextChunker`

**主要方法：**

- `chunk_text(text: str, *, source="", page_numbers=None, pdf_metadata=None, index_offset=0) -> list[TextChunk]`
  - 对任意字符串切块。
- `chunk_page(page: CleanedPage, *, source="", pdf_metadata=None, index_offset=0) -> list[TextChunk]`
  - 页面级切块，自动填充页码与元信息。
- `chunk_content(content: CleanedContent) -> list[TextChunk]`
  - 全书级切块：
  - `page_aware=True`：逐页切，`chunk_index` 在多页之间连续增长。
  - `page_aware=False`：在各页前插入 `<<<PAGE:x>>>` 标记构建 `full_text`，切完再通过位置反推 `page_numbers`，并移除标记。

内部实现：

- 递归拆分器：`RecursiveCharacterTextSplitter`
- token 拆分器：`TokenTextSplitter`（使用 tiktoken）
- `_sha256_id(text)`: 统一计算块 ID。

---

## 4. 向量库与流水线：`chroma/chroma_store.py`

**外部依赖：**

- `langchain_chroma.Chroma`
- `backend.llm.openai_client.get_embeddings()`（OpenAI Embeddings）
- 与全局配置结合：
  - `Settings.chroma_persist_dir`：Chroma 持久化目录
  - `Settings.openai_embedding_model`：嵌入模型名称

**配置与结果：**

- `IngestConfig`
  - `skip_existing`: 是否按 `chunk_id` 去重（幂等写入）
  - `embed_batch_size`: 向量化批次大小
- `IngestResult`
  - `source/total_chunks/added/skipped/collection_name`
  - `__str__` 提供友好的统计输出（例如：`[kant] kant.pdf → 总计 312 块，写入 312，跳过 0`）

**核心类：**

- `ChromaStore`

**主要职责：**

1. **一键流水线入库：**
   - `ingest_pdf(path, start_page=1, end_page=None, collection_name=None) -> IngestResult`
     - 执行：
       1. `PDFExtractor(path).extract(...)`
       2. `TextCleaner(clean_config).clean_content(...)`
       3. `TextChunker(chunk_config).chunk_content(...)`
       4. `_ingest_chunks_to_db(chunks, db, source=str(path))`
     - 具备幂等能力：开启 `skip_existing` 后，多次 ingest 同一 PDF 不会重复写入。

2. **直接写入切好的块：**
   - `ingest_chunks(chunks, collection_name=None) -> IngestResult`
     - 适合上游已自定义清洗/切块的场景。

3. **删除与维护：**
   - `delete_source(source: str) -> int`
     - 按元数据字段 `source` 从 collection 中删除所有对应文档。
   - `get_stats() -> dict`
     - 返回 `collection_name/persist_directory/total_chunks`。
   - `list_sources() -> list[str]`
     - 读取 metadatas 中的所有 `source`，去重、排序。

4. **检索与 Retriever：**
   - `similarity_search(query, k=4, filter=None) -> list[Document]`
   - `similarity_search_with_score(query, k=4, filter=None) -> list[(Document, score)]`
   - `as_retriever(**kwargs)`
     - 可以直接接入 LangChain 的 Chain / Agent。

5. **多 collection 支持：**
   - 类实例内部持有一个默认 `Chroma` 实例（`collection_name` 构造时指定）。
   - `_resolve_db(collection_name)`：
     - 若传入与默认不同的名称，则按相同 `persist_directory` 创建/加载另一个 collection。

写入时的元数据转换：

- `_chunk_to_document(chunk: TextChunk) -> Document`
  - 确保所有 `metadata` 值为 `str/int/float/bool`：
    - `page_numbers` 被序列化为 `","` 分隔字符串
    - `chunk_id/char_count/source/chunk_index/pdf_title/pdf_author` 均按原样保存。

---

## 5. 测试与可靠性

`tests/rag` 为 RAG 子系统专用测试，覆盖率约 98–100%，关键点：

- `conftest.py`
  - 动态使用 PyMuPDF 生成最小可控 PDF（3 页，英文内容 + 页眉页脚 + 页码）。
  - 提供丰富的 fixture（`PageContent` / `CleanedPage` / `TextChunk` 等）。
- 单元测试文件：
  - `test_pdf_extractor.py`：覆盖文件校验、页范围边界、元数据、结构完整性、迭代和纯文本输出。
  - `test_text_cleaner.py`：覆盖各个配置开关、启发式规则、块过滤逻辑和 `full_text` 行为。
  - `test_text_chunker.py`：覆盖 chunk 大小/重叠、ID 生成、一致性、页感知/全文模式、token 拆分等。
  - `test_chroma_store.py`：
    - 对 `Chroma` 与 Embeddings 使用 `monkeypatch` + `MagicMock` 做纯本地单元测试。
    - 覆盖入库、去重、分批写入、删除、检索、统计、多 collection 的行为。

---

### 小结

当前 `rag` 模块已经提供了：

- **清晰的数据结构**：`PageContent / PDFContent / CleanedPage / CleanedContent / TextChunk / ChunkMeta / IngestResult`
- **可配置的全链路处理**：清洗 & 切块 & 入库 都通过配置类控制，适合不同文档类型（书籍、论文、说明书）。
- **可直接嵌入 LangChain**：`ChromaStore.as_retriever` 使得 RAG 部署到对话 / Agent 流水线非常简单。
- **高测试覆盖率**：大部分逻辑都有针对性的单测与边界测试，便于你后续改动和扩展。