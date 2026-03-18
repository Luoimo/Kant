# EPUB RAG Pipeline 设计文档

**日期**：2026-03-18
**状态**：已批准，待实现
**作者**：Chloe

---

## 背景

原有 RAG 链路基于 PyMuPDF 解析 PDF 文件。由于目标书籍主要以 EPUB 格式存在，EPUB 拥有原生的章节结构、TOC 和 metadata，比 PDF 更适合做语义切分。本次完全替换 PDF 链路，改为 EPUB 支持。

---

## 目标

- 支持 EPUB 文件的全链路 RAG 入库（提取 → 清洗 → 切块 → 向量化 → 存储）
- 每个 chunk 携带准确的 `chapter_title`、`section_title`、`book_title`、`author` 元数据
- 按章节切分，章节内再做软切分
- 删除所有 PDF 相关代码，数据模型使用格式无关的中立命名

---

## 架构总览

### 数据流

```
EpubExtractor.extract()
    → BookContent（sections + toc + metadata）
        → TextCleaner.clean_content()
            → CleanedBookContent（CleanedSection 列表）
                → TextChunker.chunk_content()
                    → list[TextChunk]（含 chapter_title / section_title）
                        → ChromaStore.ingest()
```

### 文件变动

```
backend/rag/
    extracter/
        epub_extractor.py       ← 新增
        pdf_extractor.py        ← 删除
        __init__.py             ← 更新

    cleaner/
        text_cleaner.py         ← 修改（类名重命名，逻辑不变）

    chunker/
        text_chunker.py         ← 修改（字段重命名，逻辑不变）

    chroma/
        chroma_store.py         ← 修改（ingest_pdf→ingest，引用新类名）

requirements.txt                ← 新增 ebooklib、beautifulsoup4、lxml
                                   删除 pymupdf、unstructured

tests/rag/
    conftest.py                 ← 替换（EPUB fixture，移除 PDF fixture）
    test_epub_extractor.py      ← 新增（替代 test_pdf_extractor.py）
    test_text_cleaner.py        ← 修改（类名更新）
    test_text_chunker.py        ← 修改（字段名更新）
    test_chroma_store.py        ← 修改（ingest_pdf → ingest）
```

---

## 数据模型（中立命名）

### 重命名映射

| 现有名称 | 新名称 |
|---|---|
| `PDFContent` | `BookContent` |
| `PageContent` | `SectionContent` |
| `CleanedContent` | `CleanedBookContent` |
| `CleanedPage` | `CleanedSection` |
| `ChunkMeta.pdf_title` | `ChunkMeta.book_title` |
| `ChunkMeta.pdf_author` | `ChunkMeta.author` |
| `ChunkMeta.page_numbers` | `ChunkMeta.section_indices` |
| `ChunkConfig.page_aware` | `ChunkConfig.section_aware` |
| `TextChunker.chunk_page()` | `TextChunker.chunk_section()` |
| `build_page_section_map()` | `build_section_map()` |

### `SectionContent`

```python
@dataclass
class SectionContent:
    section_index: int      # 0-based，在书中的顺序
    title: str              # 章节标题（来自 EPUB TOC 或 <h1>/<h2>）
    text: str               # 纯文本
    blocks: list[dict]      # 文本块，含 is_heading / heading_text / start_offset
    source_href: str        # EPUB 内部 HTML 文件路径
```

`blocks` 中每个元素对应一个段落（`<p>`）或标题（`<h1>`~`<h3>`），结构与现有 `PageContent.blocks` 兼容。

### `BookContent`

```python
@dataclass
class BookContent:
    source: str                    # EPUB 文件路径
    total_sections: int
    metadata: dict                 # title / author / language / publisher
    sections: list[SectionContent]
    toc: list[TOCEntry]            # [(level, title, section_index)]
```

### `ChunkMeta`

```python
@dataclass
class ChunkMeta:
    source: str
    section_indices: list[int]
    chunk_index: int
    book_title: str
    author: str
    chapter_title: str
    section_title: str
```

---

## EpubExtractor

### 依赖

- `ebooklib`：解析 EPUB 结构（OPF metadata、spine、NCX/NAV TOC）
- `beautifulsoup4` + `lxml`：解析章节 HTML

### 核心流程

```
EpubExtractor.extract()
    │
    ├─ ebooklib.read_epub()
    │
    ├─ _extract_metadata()   → title / author / language / publisher（OPF）
    │
    ├─ _extract_toc()        → [(level, title, section_index)]
    │     优先读 NCX toc.ncx，fallback 读 NAV nav.xhtml
    │
    └─ 按 spine 顺序遍历 HTML 章节文件：
          ├─ BeautifulSoup 解析，跳过 <script>/<style>/<header>/<footer>/<img>
          ├─ <h1>/<h2>/<h3> → is_heading=True，直接读标签
          ├─ <p> → 普通文本块
          └─ 构建 SectionContent
```

### 标题层级

| HTML 标签 | TOC level |
|---|---|
| `<h1>` | 1（章） |
| `<h2>` | 2（节） |
| `<h3>` | 3（小节） |

### 公开接口

```python
class EpubExtractor:
    def __init__(self, path: str | Path) -> None: ...
    def extract(self) -> BookContent: ...
    def get_metadata(self) -> dict: ...
```

无需 `start/end_page` 参数，EPUB 以完整章节为单位。

---

## TextCleaner 适配

逻辑完全不变，仅做类名适配：

- 接收 `BookContent` / `SectionContent`，返回 `CleanedBookContent` / `CleanedSection`
- `build_section_map(toc, total_sections)` 替代 `build_page_section_map()`
- 删除 `page_height` / `remove_headers_footers` 启发式（EPUB 的 `<header>`/`<footer>` 在 `EpubExtractor` 解析时直接跳过）
- `CleanedSection` 保留 `chapter_title` / `section_title`，来自 TOC 映射

---

## TextChunker 适配

核心软切分逻辑不变，仅字段重命名：

- `chunk_page(page: CleanedPage)` → `chunk_section(section: CleanedSection)`
- `ChunkConfig.section_aware`（原 `page_aware`），默认 `True`
- `ChunkMeta` 字段按重命名映射更新
- `_section_titles_for_chunks()` 保留，依赖 `blocks` 的 `is_heading`/`heading_text`/`start_offset`

---

## ChromaStore 适配

```python
# 修改前
def ingest_pdf(self, path, ...) -> IngestResult:
    extractor = PDFExtractor(path)
    pdf_content = extractor.extract(start_page=..., end_page=...)

# 修改后
def ingest(self, path, ...) -> IngestResult:
    extractor = EpubExtractor(path)
    book_content = extractor.extract()
```

`_chunk_to_document()` 字段名同步更新：`pdf_title` → `book_title`，`page_numbers` → `section_indices`（序列化方式不变，逗号分隔）。

---

## 测试策略

### 原则

不依赖真实 EPUB 文件，所有 fixture 用 `ebooklib` 在内存中构造。

### Fixture

```python
@pytest.fixture(scope="session")
def sample_epub_path(tmp_path_factory):
    # 含 3 章节 + NCX TOC 的最小 EPUB
    # 每章有 <h1> 标题 + 若干 <p> 段落
    # OPF 含 title / author
```

### 关键测试点

| 测试文件 | 验证内容 |
|---|---|
| `test_epub_extractor.py` | metadata 读取、TOC 条目数量和层级、`<h1>` 标记为 `is_heading` |
| `test_text_cleaner.py` | `chapter_title`/`section_title` 正确从 TOC 映射到 `CleanedSection` |
| `test_text_chunker.py` | `section_indices` 正确、`chapter_title` 随 chunk 携带、软切分不破坏中文句子 |
| `test_chroma_store.py` | `ingest()` 写入正确数量 chunk，metadata 含 `book_title` 和 `chapter_title` |

---

## 依赖变更

### 新增

```
ebooklib>=0.18
beautifulsoup4>=4.12
lxml>=5.0
```

### 删除

```
pymupdf
unstructured
python-docx
```
