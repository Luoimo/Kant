# _*_ coding:utf-8 _*_
# @Time:2026/3/10
# @Author:Chloe
# @File:conftest.py
# @Project:Kant
"""
测试共享 fixture。

样本 PDF 由 PyMuPDF 在内存中生成，不依赖外部文件。
所有 PageContent / CleanedContent / TextChunk 均为纯 Python 数据对象，
无需网络或 GPU。
"""
from __future__ import annotations

import pytest
import fitz  # pymupdf

from backend.rag.extracter.pdf_extractor import PageContent, PDFContent
from backend.rag.cleaner.text_cleaner import CleanedPage, CleanedContent
from backend.rag.chunker.text_chunker import TextChunk, ChunkMeta

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
SAMPLE_TITLE = "Test Book"
SAMPLE_AUTHOR = "Test Author"

# 用于内存 fixture（PageContent / CleanedPage）的中文文本
PAGE_TEXTS = [
    "第一章 纯粹理性批判导言。\n\n"
    "人类理性在其知识的某一门类中具有一种特殊命运，"
    "它被种种它无法拒绝的问题所困扰，因为这些问题是由理性本性本身"
    "向它提出的，但它又无法回答，因为这些问题超越了人类理性的一切能力。",

    "第二章 先验感性论。\n\n"
    "通过直觉，对象被给予我们，并且只有这样，对象才与我们的"
    "思维相关联。直觉只在对象被给予我们时才得以发生，但这对我们"
    "人类来说，至少是只有对象以某种方式触动我们的心灵才有可能。",

    "第三章 先验逻辑论。\n\n"
    "我们的知识起源于心灵的两个基本来源：第一个是接受表象的能力，"
    "即感受性；第二个是通过这些表象认识对象的能力，即自发性。"
    "通过感受性，对象被给予我们；通过自发性，对象被我们思维。",
]

# 用于生成样本 PDF 的纯 ASCII 文本
# PyMuPDF 默认 Base-14 字体不含 CJK 字形，必须使用 ASCII 文本避免输出乱码
_PDF_ASCII_TEXTS = [
    "Chapter 1. Critique of Pure Reason.\n\n"
    "Human reason has a peculiar fate in one domain of knowledge: "
    "it is burdened with questions it cannot dismiss, because they arise "
    "from the very nature of reason itself, yet it cannot answer them, "
    "because they surpass every faculty of human reason.",

    "Chapter 2. Transcendental Aesthetic.\n\n"
    "In the transcendental aesthetic we shall first isolate sensibility by "
    "separating off everything that the understanding thinks through its concepts, "
    "so that nothing but empirical intuition shall remain.",

    "Chapter 3. Transcendental Logic.\n\n"
    "Our knowledge springs from two fundamental sources of the mind: "
    "the first is the capacity of receiving representations, "
    "the second is the power of knowing an object through those representations.",
]


# ---------------------------------------------------------------------------
# PDF 文件 fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_pdf_path(tmp_path_factory):
    """生成一个包含 3 页文本的最小 PDF 文件，供整个测试会话共享。"""
    pdf_path = tmp_path_factory.mktemp("pdf") / "sample.pdf"
    doc = fitz.open()
    for i, text in enumerate(_PDF_ASCII_TEXTS):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), text, fontsize=11)
        # 在页面顶部写页眉，底部写页码（用于测试页眉页脚过滤）
        page.insert_text((72, 20), f"Header - {SAMPLE_TITLE}", fontsize=9)
        page.insert_text((280, 820), str(i + 1), fontsize=9)
    doc.set_metadata({"title": SAMPLE_TITLE, "author": SAMPLE_AUTHOR})
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------------
# PageContent fixtures
# ---------------------------------------------------------------------------

def _make_page_content(
    page_number: int = 1,
    text: str | None = None,
    blocks: list[dict] | None = None,
    width: float = 595.0,
    height: float = 842.0,
) -> PageContent:
    text = text or PAGE_TEXTS[0]
    blocks = blocks or [
        {
            "bbox": (72.0, 100.0, 500.0, 300.0),
            "text": text,
            "block_no": 0,
            "block_type": 0,
        }
    ]
    return PageContent(
        page_number=page_number,
        text=text,
        blocks=blocks,
        images=[],
        width=width,
        height=height,
    )


@pytest.fixture
def sample_page_content():
    return _make_page_content()


@pytest.fixture
def sample_pdf_content():
    pages = [_make_page_content(i + 1, PAGE_TEXTS[i]) for i in range(3)]
    return PDFContent(
        source="data/books/sample.pdf",
        total_pages=3,
        metadata={
            "title": SAMPLE_TITLE,
            "author": SAMPLE_AUTHOR,
            "subject": "",
            "keywords": "",
            "creator": "",
            "producer": "",
            "creation_date": "",
            "mod_date": "",
            "page_count": 3,
        },
        pages=pages,
    )


@pytest.fixture
def page_with_image_blocks():
    """含图片块（block_type=1）的页面，清洗后应被过滤掉。"""
    return _make_page_content(
        blocks=[
            {"bbox": (72, 100, 500, 300), "text": PAGE_TEXTS[0], "block_no": 0, "block_type": 0},
            {"bbox": (72, 310, 500, 500), "text": "", "block_no": 1, "block_type": 1},  # 图片块
        ]
    )


@pytest.fixture
def page_with_short_blocks():
    """含过短文本块的页面，清洗后应被过滤掉（低于 min_block_chars）。"""
    return _make_page_content(
        blocks=[
            {"bbox": (72, 100, 500, 300), "text": PAGE_TEXTS[0], "block_no": 0, "block_type": 0},
            {"bbox": (72, 310, 200, 330), "text": "ab", "block_no": 1, "block_type": 0},  # 太短
        ]
    )


@pytest.fixture
def page_with_header_footer():
    """含页眉（y0 < 20）和页脚（y1 > 822）块的页面。"""
    return _make_page_content(
        height=842.0,
        blocks=[
            # 正文块
            {"bbox": (72, 100, 500, 700), "text": PAGE_TEXTS[0], "block_no": 0, "block_type": 0},
            # 页眉（y0=5，紧贴顶部）
            {"bbox": (72, 5, 500, 18), "text": f"Header - {SAMPLE_TITLE}", "block_no": 1, "block_type": 0},
            # 页脚（y1=838，紧贴底部）
            {"bbox": (280, 825, 320, 838), "text": "1", "block_no": 2, "block_type": 0},
        ],
    )


# ---------------------------------------------------------------------------
# CleanedPage / CleanedContent fixtures
# ---------------------------------------------------------------------------

def _make_cleaned_page(page_number: int = 1, text: str | None = None) -> CleanedPage:
    src = _make_page_content(page_number, text)
    txt = text if text is not None else PAGE_TEXTS[0]
    return CleanedPage(
        page_number=page_number,
        text=txt,
        blocks=[{"bbox": (72, 100, 500, 300), "text": txt, "block_no": 0, "block_type": 0}],
        width=595.0,
        height=842.0,
        source_page=src,
    )


@pytest.fixture
def sample_cleaned_page():
    return _make_cleaned_page()


@pytest.fixture
def sample_cleaned_content():
    pages = [_make_cleaned_page(i + 1, PAGE_TEXTS[i]) for i in range(3)]
    return CleanedContent(
        source="data/books/sample.pdf",
        total_pages=3,
        metadata={"title": SAMPLE_TITLE, "author": SAMPLE_AUTHOR, "page_count": 3},
        pages=pages,
    )


# ---------------------------------------------------------------------------
# TextChunk fixtures
# ---------------------------------------------------------------------------

def make_chunk(
    text: str = "This is a sample chunk for testing purposes.",
    source: str = "sample.pdf",
    page_numbers: list[int] | None = None,
    chunk_index: int = 0,
) -> TextChunk:
    from backend.rag.chunker.text_chunker import _sha256_id
    t = text
    return TextChunk(
        chunk_id=_sha256_id(t),
        text=t,
        char_count=len(t),
        metadata=ChunkMeta(
            source=source,
            page_numbers=page_numbers if page_numbers is not None else [1],
            chunk_index=chunk_index,
            pdf_title=SAMPLE_TITLE,
            pdf_author=SAMPLE_AUTHOR,
        ),
    )


@pytest.fixture
def sample_chunks():
    return [
        make_chunk(f"Chunk number {i} contains important philosophical content.", chunk_index=i)
        for i in range(5)
    ]
