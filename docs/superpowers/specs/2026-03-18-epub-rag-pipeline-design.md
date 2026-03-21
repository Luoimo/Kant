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
        text_cleaner.py         ← 修改（类名重命名，CleanConfig 清理）

    chunker/
        text_chunker.py         ← 修改（字段重命名，逻辑不变）

    chroma/
        chroma_store.py         ← 修改（ingest_pdf→ingest，引用新类名，更新文档注释）

requirements.txt                ← 新增 ebooklib、beautifulsoup4、lxml
                                   移除 pymupdf、python-docx
                                   保留 unstructured（TextCleaner 仍使用其清洗函数）

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
| `TextChunker._chunk_page_aware()` | `TextChunker._chunk_section_aware()` |
| `build_page_section_map()` | `build_section_map()` |

### `TOCEntry` 类型（重新定义）

```python
# 格式：(层级 1-based, 标题, section_index 0-based)
TOCEntry = tuple[int, str, int]
```

**注意**：原有 `TOCEntry` 第三元素为 PDF 页码（1-based），现改为 EPUB spine 的 `section_index`（0-based）。`build_section_map()` 的匹配逻辑改为 `entry_section_index <= current_section_index`。

### `SectionContent`

```python
@dataclass
class SectionContent:
    section_index: int      # 0-based，spine 顺序
    title: str              # 章节标题（来自 TOC 或首个 <h1>/<h2>）
    text: str               # 纯文本（空章节为 ""）
    blocks: list[dict]      # 文本块，含 is_heading / heading_text / start_offset
    source_href: str        # EPUB 内部 HTML 文件路径（不含 fragment）
```

`blocks` 中每个元素结构：

```python
{
    "text": str,
    "block_no": int,
    "block_type": 0,          # EPUB 无图片块，始终为 0
    "is_heading": bool,       # <h1>~<h3> 为 True
    "heading_text": str,      # is_heading=True 时填充
    "start_offset": int,      # 在本章 text 中的字符偏移（TextCleaner 计算）
    "end_offset": int,
    # 无 bbox 字段（EPUB 无页面坐标）
}
```

### `BookContent`

```python
@dataclass
class BookContent:
    source: str                    # EPUB 文件路径
    total_sections: int            # spine 中有效章节数
    metadata: dict                 # title / author / language / publisher
    sections: list[SectionContent]
    toc: list[TOCEntry]            # [(level, title, section_index)]；无 TOC 时为 []
```

### `CleanedSection`

```python
@dataclass
class CleanedSection:
    section_index: int
    text: str
    blocks: list[dict]             # 清洗后文本块（含 start_offset / end_offset）
    source_section: SectionContent # 原始 SectionContent 引用
    chapter_title: str = ""        # TOC 层级 1 标题
    section_title: str = ""        # TOC 最细层级标题
```

**移除**：`width`、`height`（EPUB 无页面几何）、`source_page`（改为 `source_section`）。

### `CleanedBookContent`

```python
@dataclass
class CleanedBookContent:
    source: str
    total_sections: int
    metadata: dict
    sections: list[CleanedSection]

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections if s.text.strip())
```

### `ChunkMeta`

```python
@dataclass
class ChunkMeta:
    source: str
    section_indices: list[int]   # 原 page_numbers
    chunk_index: int
    book_title: str              # 原 pdf_title
    author: str                  # 原 pdf_author
    chapter_title: str
    section_title: str
```

### `TextChunk.to_dict()`

同步更新键名：

```python
def to_dict(self) -> dict:
    return {
        "chunk_id": self.chunk_id,
        "text": self.text,
        "char_count": self.char_count,
        "source": self.metadata.source,
        "section_indices": self.metadata.section_indices,   # 原 page_numbers
        "chunk_index": self.metadata.chunk_index,
        "book_title": self.metadata.book_title,             # 原 pdf_title
        "author": self.metadata.author,                     # 原 pdf_author
        "chapter_title": self.metadata.chapter_title,
        "section_title": self.metadata.section_title,
    }
```

---

## EpubExtractor

### 依赖

- `ebooklib`：解析 EPUB 结构（OPF metadata、spine、NCX/NAV TOC）
- `beautifulsoup4` + `lxml`：解析章节 HTML

### href → section_index 映射策略

EPUB TOC（NCX/NAV）的条目以 href 引用章节文件（可能含 fragment，如 `chapter01.xhtml#section2`）。映射步骤：

1. 按 spine 顺序枚举所有 HTML 文件，建立 `{href_without_fragment: section_index}` 字典
2. 提取 TOC 条目的 href，去掉 fragment（`#` 之后的部分），查表得到 `section_index`
3. 若 TOC href 无法匹配任何 spine 文件（EPUB 格式错误），该条目跳过

### 核心流程

```
EpubExtractor.extract()
    │
    ├─ ebooklib.read_epub()
    │
    ├─ _build_href_index()     → {href: section_index} 映射表
    │
    ├─ _extract_metadata()     → title / author / language / publisher（OPF）
    │                             任意字段缺失时返回空字符串
    │
    ├─ _extract_toc()          → list[TOCEntry]
    │     优先 NCX toc.ncx，fallback NAV nav.xhtml
    │     两者均无或均为空 → 返回 []（下游 build_section_map 可处理空 TOC）
    │
    └─ 按 spine 顺序遍历 HTML 章节文件：
          ├─ BeautifulSoup 解析，跳过 <script>/<style>/<header>/<footer>/<img>
          ├─ <h1>/<h2>/<h3> → is_heading=True，heading_text = 标签文本
          ├─ <p> → 普通文本块，is_heading=False
          ├─ 无任何 <p>/<h*> 内容的章节 → SectionContent(text="", blocks=[])，仍保留
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

### `CleanConfig` 变化

移除与 EPUB 无关的选项：

```python
# 移除以下字段（EPUB 无页面几何，header/footer 由 EpubExtractor 在 HTML 层过滤）
remove_headers_footers: bool   # 删除
header_footer_lines: int       # 删除
```

其余选项（`extra_whitespace`、`bullets`、`dashes`、`fix_hyphenation` 等）继续适用于 EPUB 文本。

### 接口变化

- 接收 `BookContent` / `SectionContent`，返回 `CleanedBookContent` / `CleanedSection`
- `clean_content(content: BookContent) -> CleanedBookContent`
- `clean_section(section: SectionContent) -> CleanedSection`（原 `clean_page`）
- `build_section_map(toc, total_sections)` 替代 `build_page_section_map()`
  - 无 TOC 时（`toc=[]`）返回 `{i: ("", "") for i in range(total_sections)}`

### `_clean_blocks` 变化

移除 `page_height` 参数和 bbox 相关的 header/footer 判断（`_is_header_footer` 方法删除）。`unstructured` 的清洗函数继续使用。

---

## TextChunker 适配

核心软切分逻辑不变，字段和方法重命名：

| 现有 | 修改后 |
|---|---|
| `chunk_page(page: CleanedPage, ...)` | `chunk_section(section: CleanedSection, ...)` |
| `chunk_text(..., pdf_metadata=...)` | `chunk_text(..., book_metadata=...)` |
| `_chunk_page_aware(content)` | `_chunk_section_aware(content)` |
| `_chunk_fulltext()` 内 `page.page_number` | `section.section_index` |
| `_chunk_fulltext()` 内 `page_titles` dict | `section_titles` dict，以 `section_index` 为 key |
| `content.pages` | `content.sections` |
| `ChunkConfig.page_aware` | `ChunkConfig.section_aware` |
| 内部方法参数 `pdf_title`/`pdf_author`/`page_numbers` | `book_title`/`author`/`section_indices` |
| `ChunkMeta` 字段 | 按重命名映射更新 |

`_section_titles_for_chunks()` 和软切分（`_soft_split`、`_soft_split_positions`）保留，逻辑不变。

**注意**：`clean_section()` 单独调用时 `chapter_title`/`section_title` 始终为空字符串；这两个字段只有通过 `clean_content()` 调用（内部执行 `build_section_map()`）才能被正确填充。

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

类注释、docstring 同步更新，移除所有 PDF 相关描述（"PDF 向量化存储管理类" 等）。

---

## 边界情况处理

| 情况 | 处理方式 |
|---|---|
| EPUB 无 NCX/NAV TOC | `toc=[]`，`chapter_title`/`section_title` 全为空字符串 |
| TOC href 无法匹配 spine | 该 TOC 条目跳过 |
| 章节 HTML 解析后无文本 | 保留 `SectionContent(text="", blocks=[])`，TextChunker 跳过空章节（`not section.text.strip()` → return `[]`） |
| metadata 字段缺失 | 对应字段返回空字符串，不报错 |
| 单章节 EPUB | `section_indices=[0]`，正常处理 |
| 同一 HTML 被多个 TOC 条目引用 | 取 href 匹配的 section_index，多个条目共享同一 index，层级由 level 字段区分 |

---

## 测试策略

### 原则

不依赖真实 EPUB 文件，所有 fixture 用 `ebooklib` 在内存中构造，风格与现有 `conftest.py` 一致。

### Fixture 示例（`conftest.py`）

```python
@pytest.fixture(scope="session")
def sample_epub_path(tmp_path_factory):
    import ebooklib
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_title("Test Book")
    book.add_author("Test Author")
    book.set_language("zh")

    chapters = []
    for i, (ch_title, body) in enumerate([
        ("第一章 导言", "<h1>第一章 导言</h1><p>人类理性在其知识的某一门类中具有一种特殊命运。</p>"),
        ("第二章 先验感性论", "<h1>第二章 先验感性论</h1><p>通过直觉，对象被给予我们。</p>"),
        ("第三章 先验逻辑论", "<h1>第三章 先验逻辑论</h1><p>我们的知识起源于心灵的两个基本来源。</p>"),
    ]):
        c = epub.EpubHtml(title=ch_title, file_name=f"chap{i+1}.xhtml", lang="zh")
        c.content = f"<html><body>{body}</body></html>"
        book.add_item(c)
        chapters.append(c)

    # TOC：嵌套结构，第一章含一个二级小节，用于验证 level 字段提取
    book.toc = [
        (epub.Section("第一章 导言"), [
            epub.Link("chap1.xhtml", "第一章 导言", "chap1"),
            epub.Link("chap1.xhtml#sec1", "1.1 理性的命运", "sec1"),
        ]),
        epub.Link("chap2.xhtml", "第二章 先验感性论", "chap2"),
        epub.Link("chap3.xhtml", "第三章 先验逻辑论", "chap3"),
    ]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    # 注意：spine 不含 "nav"，避免 nav 占据 section_index=0 导致章节索引偏移
    # EpubExtractor 在构建 href→section_index 映射时只计入 EpubHtml 类型的 spine 条目
    book.spine = chapters

    path = tmp_path_factory.mktemp("epub") / "sample.epub"
    epub.write_epub(str(path), book)
    return path
```

**spine 索引规则**：`EpubExtractor` 构建 `{href: section_index}` 时，仅对 `EpubHtml` 类型的 spine 条目计数，跳过 `EpubNcx`/`EpubNav` 等非内容条目。这样 `chap1.xhtml` → `section_index=0`，`chap2.xhtml` → `section_index=1`，以此类推。

### 关键测试点

| 测试文件 | 验证内容 |
|---|---|
| `test_epub_extractor.py` | metadata 读取、TOC 条目数量和层级、`<h1>` 标记为 `is_heading`、空 TOC fallback、空章节处理 |
| `test_text_cleaner.py` | `chapter_title`/`section_title` 正确从 TOC 映射到 `CleanedSection`、无 TOC 时全为空字符串 |
| `test_text_chunker.py` | `section_indices` 正确、`chapter_title` 随 chunk 携带、软切分不破坏中文句子 |
| `test_chroma_store.py` | `ingest()` 写入正确数量 chunk，metadata 含 `book_title` 和 `chapter_title`，`section_indices` 正确序列化 |

---

## 依赖变更

### 新增

```
ebooklib>=0.18
beautifulsoup4>=4.12
lxml>=5.0
```

### 移除

```
pymupdf
python-docx
```

### 保留

```
unstructured    # TextCleaner 仍使用其文本清洗函数（clean、clean_bullets 等）
```
