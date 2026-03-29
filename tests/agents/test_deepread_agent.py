from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document

from backend.agents.deepread_agent import DeepReadAgent, DeepReadConfig


def _make_doc(content: str, source: str = "data/books/kant.epub", chapter: str = "第一章") -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "section_title": chapter, "book_title": "纯粹理性批判"},
    )


@pytest.fixture
def agent():
    mock_store = MagicMock()
    mock_llm = MagicMock()
    a = DeepReadAgent(store=mock_store, llm=mock_llm, config=DeepReadConfig(k=6, fetch_k=20))
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [_make_doc("先验统觉是…")]
    a._retriever = mock_retriever
    return a, mock_retriever


def test_search_books_current_book_applies_source_filter(agent):
    a, mock_retriever = agent
    a._search_books_impl(
        search_query="先验统觉",
        scope="current_book",
        book_source="data/books/kant.epub",
        chapter=None,
    )
    mock_retriever.search.assert_called_once_with(
        "先验统觉", filter={"source": "data/books/kant.epub"}
    )


def test_search_books_all_books_no_filter(agent):
    a, mock_retriever = agent
    a._search_books_impl(
        search_query="自由意志",
        scope="all_books",
        book_source="data/books/kant.epub",
        chapter=None,
    )
    mock_retriever.search.assert_called_once_with("自由意志", filter=None)


def test_search_books_chapter_prepends_to_query(agent):
    a, mock_retriever = agent
    a._search_books_impl(
        search_query="总结",
        scope="current_book",
        book_source="data/books/kant.epub",
        chapter="先验感性论",
    )
    call_args = mock_retriever.search.call_args
    assert "先验感性论" in call_args[0][0]


def test_search_books_no_book_source_current_book_no_filter(agent):
    a, mock_retriever = agent
    a._search_books_impl(
        search_query="什么是哲学",
        scope="current_book",
        book_source=None,
        chapter=None,
    )
    mock_retriever.search.assert_called_once_with("什么是哲学", filter=None)
