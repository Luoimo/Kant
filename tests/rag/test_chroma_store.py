# _*_ coding:utf-8 _*_
"""
ChromaStore 单元测试。

所有对 Chroma / OpenAI Embeddings 的调用均通过 monkeypatch + MagicMock 拦截，
测试不依赖网络、不写磁盘数据库（persist_directory 使用 tmp_path）。

覆盖：
- _chunk_to_document 转换正确性
- ingest_chunks 写入流程（正常 / 幂等去重 / 空列表）
- ingest_chunks 分批写入（embed_batch_size）
- delete_source 触发正确的 collection.delete
- similarity_search / similarity_search_with_score 代理调用
- as_retriever 返回可调用对象
- get_stats 返回正确字段
- list_sources 去重排序
- IngestResult.__str__ 格式
- ingest 全流水线（pipeline 步骤均 mock）
- _resolve_db 多 collection 支持
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from langchain_core.documents import Document

from backend.rag.chroma.chroma_store import (
    ChromaStore,
    IngestConfig,
    IngestResult,
)
from backend.rag.chunker.text_chunker import ChunkMeta, TextChunk
from tests.rag.conftest import make_chunk, SAMPLE_TITLE, SAMPLE_AUTHOR


# ---------------------------------------------------------------------------
# 共享 fixture：带 mock 的 ChromaStore
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_collection():
    col = MagicMock()
    col.name = "test_col"
    col.count.return_value = 0
    col.get.return_value = {"ids": [], "metadatas": []}
    return col


@pytest.fixture
def mock_db(mock_collection):
    db = MagicMock()
    db._collection = mock_collection
    db.similarity_search.return_value = []
    db.similarity_search_with_score.return_value = []
    db.as_retriever.return_value = MagicMock()
    return db


@pytest.fixture
def store(tmp_path, mock_db, monkeypatch):
    """
    初始化 ChromaStore，完全 mock 掉 get_embeddings 和 Chroma 构造函数。
    """
    monkeypatch.setattr(
        "backend.rag.chroma.chroma_store.get_embeddings",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "backend.rag.chroma.chroma_store.Chroma",
        lambda **kwargs: mock_db,
    )
    s = ChromaStore(
        collection_name="test_col",
        persist_directory=str(tmp_path),
    )
    return s


# ---------------------------------------------------------------------------
# IngestResult
# ---------------------------------------------------------------------------

class TestIngestResult:

    def test_str_contains_filename(self):
        r = IngestResult(
            source="/path/to/kant.epub",
            total_chunks=100,
            added=90,
            skipped=10,
            collection_name="kant",
        )
        assert "kant.epub" in str(r)

    def test_str_contains_counts(self):
        r = IngestResult(
            source="kant.epub",
            total_chunks=100,
            added=90,
            skipped=10,
            collection_name="kant",
        )
        s = str(r)
        assert "100" in s
        assert "90" in s
        assert "10" in s

    def test_str_contains_collection_name(self):
        r = IngestResult(
            source="kant.epub",
            total_chunks=5,
            added=5,
            skipped=0,
            collection_name="mylib",
        )
        assert "mylib" in str(r)


# ---------------------------------------------------------------------------
# _chunk_to_document
# ---------------------------------------------------------------------------

class TestChunkToDocument:

    def test_page_content_equals_chunk_text(self):
        chunk = make_chunk(text="Hello philosophical world.", section_indices=[3, 4])
        doc = ChromaStore._chunk_to_document(chunk)
        assert doc.page_content == chunk.text

    def test_metadata_source(self):
        chunk = make_chunk(source="kant.epub")
        doc = ChromaStore._chunk_to_document(chunk)
        assert doc.metadata["source"] == "kant.epub"

    def test_section_indices_serialized_as_string(self):
        chunk = make_chunk(section_indices=[1, 2, 3])
        doc = ChromaStore._chunk_to_document(chunk)
        assert doc.metadata["section_indices"] == "1,2,3"

    def test_single_index_serialized(self):
        chunk = make_chunk(section_indices=[7])
        doc = ChromaStore._chunk_to_document(chunk)
        assert doc.metadata["section_indices"] == "7"

    def test_empty_section_indices(self):
        chunk = make_chunk(section_indices=[])
        doc = ChromaStore._chunk_to_document(chunk)
        assert doc.metadata["section_indices"] == ""

    def test_metadata_keys_present(self):
        chunk = make_chunk()
        doc = ChromaStore._chunk_to_document(chunk)
        for key in ("chunk_id", "char_count", "source", "section_indices",
                    "chunk_index", "book_title", "author",
                    "chapter_title", "section_title"):
            assert key in doc.metadata

    def test_chunk_id_in_metadata(self):
        chunk = make_chunk()
        doc = ChromaStore._chunk_to_document(chunk)
        assert doc.metadata["chunk_id"] == chunk.chunk_id

    def test_char_count_in_metadata(self):
        chunk = make_chunk(text="hello world")
        doc = ChromaStore._chunk_to_document(chunk)
        assert doc.metadata["char_count"] == chunk.char_count

    def test_book_title_author_in_metadata(self):
        chunk = make_chunk()
        doc = ChromaStore._chunk_to_document(chunk)
        assert doc.metadata["book_title"] == SAMPLE_TITLE
        assert doc.metadata["author"] == SAMPLE_AUTHOR

    def test_all_metadata_values_are_primitive(self):
        """Chroma 只接受 str/int/float/bool 类型的元数据值。"""
        chunk = make_chunk(section_indices=[1, 2])
        doc = ChromaStore._chunk_to_document(chunk)
        for v in doc.metadata.values():
            assert isinstance(v, (str, int, float, bool)), \
                f"元数据值类型不合法：{type(v)}"


# ---------------------------------------------------------------------------
# ingest_chunks
# ---------------------------------------------------------------------------

class TestIngestChunks:

    def test_returns_ingest_result(self, store, sample_chunks):
        result = store.ingest_chunks(sample_chunks)
        assert isinstance(result, IngestResult)

    def test_added_equals_total_when_no_existing(self, store, sample_chunks):
        result = store.ingest_chunks(sample_chunks)
        assert result.added == len(sample_chunks)
        assert result.skipped == 0
        assert result.total_chunks == len(sample_chunks)

    def test_add_documents_called(self, store, mock_db, sample_chunks):
        store.ingest_chunks(sample_chunks)
        assert mock_db.add_documents.called

    def test_ids_passed_to_add_documents(self, store, mock_db, sample_chunks):
        store.ingest_chunks(sample_chunks)
        _, kwargs = mock_db.add_documents.call_args
        assert set(kwargs["ids"]) == {c.chunk_id for c in sample_chunks}

    def test_empty_chunks_returns_zero_added(self, store):
        result = store.ingest_chunks([])
        assert result.added == 0
        assert result.total_chunks == 0

    def test_source_in_result(self, store, sample_chunks):
        result = store.ingest_chunks(sample_chunks)
        assert result.source == sample_chunks[0].metadata.source

    def test_collection_name_in_result(self, store, sample_chunks):
        result = store.ingest_chunks(sample_chunks)
        assert result.collection_name == "test_col"


# ---------------------------------------------------------------------------
# ingest_chunks —— skip_existing 去重
# ---------------------------------------------------------------------------

class TestIngestChunksDedup:

    def test_all_skipped_when_all_existing(self, store, mock_collection, sample_chunks):
        existing_ids = [c.chunk_id for c in sample_chunks]
        mock_collection.get.return_value = {"ids": existing_ids}

        result = store.ingest_chunks(sample_chunks)
        assert result.skipped == len(sample_chunks)
        assert result.added == 0

    def test_partial_skip(self, store, mock_collection, sample_chunks):
        # 前两个 chunk 已存在
        existing_ids = [c.chunk_id for c in sample_chunks[:2]]
        mock_collection.get.return_value = {"ids": existing_ids}

        result = store.ingest_chunks(sample_chunks)
        assert result.skipped == 2
        assert result.added == len(sample_chunks) - 2

    def test_skip_existing_false_ignores_dedup(self, store, mock_collection, sample_chunks):
        store.ingest_config = IngestConfig(skip_existing=False)
        existing_ids = [c.chunk_id for c in sample_chunks]
        mock_collection.get.return_value = {"ids": existing_ids}

        result = store.ingest_chunks(sample_chunks)
        assert result.skipped == 0
        assert result.added == len(sample_chunks)


# ---------------------------------------------------------------------------
# ingest_chunks —— 分批写入
# ---------------------------------------------------------------------------

class TestIngestChunksBatching:

    def test_batched_add_documents_calls(self, store, mock_db, mock_collection):
        # embed_batch_size=2，5 个 chunk → 3 次调用
        store.ingest_config = IngestConfig(embed_batch_size=2)
        mock_collection.get.return_value = {"ids": []}
        chunks = [
            make_chunk(f"Chunk {i} content " * 3, chunk_index=i)
            for i in range(5)
        ]
        store.ingest_chunks(chunks)
        assert mock_db.add_documents.call_count == 3

    def test_each_batch_ids_are_correct(self, store, mock_db, mock_collection):
        store.ingest_config = IngestConfig(embed_batch_size=2)
        mock_collection.get.return_value = {"ids": []}
        chunks = [make_chunk(f"Chunk {i}", chunk_index=i) for i in range(4)]
        store.ingest_chunks(chunks)
        all_ids: list[str] = []
        for c in mock_db.add_documents.call_args_list:
            all_ids.extend(c.kwargs["ids"])
        assert set(all_ids) == {c.chunk_id for c in chunks}


# ---------------------------------------------------------------------------
# delete_source
# ---------------------------------------------------------------------------

class TestDeleteSource:

    def test_returns_count_of_deleted(self, store, mock_collection):
        mock_collection.get.return_value = {"ids": ["id1", "id2", "id3"]}
        count = store.delete_source("kant.epub")
        assert count == 3

    def test_collection_delete_called_with_ids(self, store, mock_collection):
        mock_collection.get.return_value = {"ids": ["id1", "id2"]}
        store.delete_source("kant.epub")
        mock_collection.delete.assert_called_once_with(ids=["id1", "id2"])

    def test_no_delete_when_no_matching_ids(self, store, mock_collection):
        mock_collection.get.return_value = {"ids": []}
        count = store.delete_source("nonexistent.epub")
        assert count == 0
        mock_collection.delete.assert_not_called()

    def test_where_filter_uses_source(self, store, mock_collection):
        mock_collection.get.return_value = {"ids": []}
        store.delete_source("specific_source.epub")
        mock_collection.get.assert_called_with(where={"source": "specific_source.epub"})


# ---------------------------------------------------------------------------
# similarity_search
# ---------------------------------------------------------------------------

class TestSimilaritySearch:

    def test_delegates_to_db(self, store, mock_db):
        expected = [Document(page_content="result")]
        mock_db.similarity_search.return_value = expected
        result = store.similarity_search("query", k=3)
        assert result is expected

    def test_k_passed_through(self, store, mock_db):
        mock_db.similarity_search.return_value = []
        store.similarity_search("q", k=7)
        mock_db.similarity_search.assert_called_once_with("q", k=7, filter=None)

    def test_filter_passed_through(self, store, mock_db):
        mock_db.similarity_search.return_value = []
        flt = {"book_title": "Critique"}
        store.similarity_search("q", filter=flt)
        mock_db.similarity_search.assert_called_once_with("q", k=4, filter=flt)


# ---------------------------------------------------------------------------
# similarity_search_with_score
# ---------------------------------------------------------------------------

class TestSimilaritySearchWithScore:

    def test_delegates_to_db(self, store, mock_db):
        expected = [(Document(page_content="res"), 0.12)]
        mock_db.similarity_search_with_score.return_value = expected
        result = store.similarity_search_with_score("query", k=2)
        assert result is expected

    def test_k_and_filter_passed_through(self, store, mock_db):
        mock_db.similarity_search_with_score.return_value = []
        flt = {"source": "kant.epub"}
        store.similarity_search_with_score("q", k=5, filter=flt)
        mock_db.similarity_search_with_score.assert_called_once_with(
            "q", k=5, filter=flt
        )


# ---------------------------------------------------------------------------
# as_retriever
# ---------------------------------------------------------------------------

class TestAsRetriever:

    def test_returns_object(self, store, mock_db):
        retriever = store.as_retriever()
        assert retriever is not None

    def test_delegates_kwargs_to_db(self, store, mock_db):
        store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        mock_db.as_retriever.assert_called_once_with(
            search_type="mmr", search_kwargs={"k": 5}
        )


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:

    def test_returns_dict(self, store, mock_collection):
        mock_collection.count.return_value = 42
        stats = store.get_stats()
        assert isinstance(stats, dict)

    def test_total_chunks_field(self, store, mock_collection):
        mock_collection.count.return_value = 42
        assert store.get_stats()["total_chunks"] == 42

    def test_collection_name_field(self, store):
        stats = store.get_stats()
        assert stats["collection_name"] == "test_col"

    def test_persist_directory_field(self, store, tmp_path):
        stats = store.get_stats()
        assert stats["persist_directory"] == str(tmp_path)


# ---------------------------------------------------------------------------
# list_sources
# ---------------------------------------------------------------------------

class TestListSources:

    def test_returns_sorted_unique_sources(self, store, mock_collection):
        mock_collection.get.return_value = {
            "metadatas": [
                {"source": "b.epub"},
                {"source": "a.epub"},
                {"source": "b.epub"},  # 重复
                {"source": "c.epub"},
            ]
        }
        sources = store.list_sources()
        assert sources == ["a.epub", "b.epub", "c.epub"]

    def test_empty_source_excluded(self, store, mock_collection):
        mock_collection.get.return_value = {
            "metadatas": [{"source": ""}, {"source": "kant.epub"}]
        }
        sources = store.list_sources()
        assert "" not in sources
        assert "kant.epub" in sources

    def test_empty_collection_returns_empty_list(self, store, mock_collection):
        mock_collection.get.return_value = {"metadatas": []}
        assert store.list_sources() == []

    def test_none_metadatas_handled(self, store, mock_collection):
        mock_collection.get.return_value = {"metadatas": None}
        assert store.list_sources() == []


# ---------------------------------------------------------------------------
# ingest —— 全流水线（pipeline 步骤 mock）
# ---------------------------------------------------------------------------

class TestIngest:

    def test_calls_pipeline_in_order(self, store, tmp_path, monkeypatch):
        """验证 ingest 按顺序调用 EpubExtractor → TextCleaner → TextChunker。"""
        call_order: list[str] = []

        mock_book_content = MagicMock()
        mock_book_content.sections = [MagicMock()]
        mock_book_content.metadata = {"title": "Test Book", "author": "Test Author"}

        mock_cleaned = MagicMock()
        mock_cleaned.sections = [MagicMock()]

        mock_chunks = [make_chunk(f"chunk {i}", chunk_index=i) for i in range(3)]

        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract.side_effect = \
            lambda: (call_order.append("extract"), mock_book_content)[1]

        mock_cleaner_instance = MagicMock()
        mock_cleaner_instance.clean_content.side_effect = \
            lambda c: (call_order.append("clean"), mock_cleaned)[1]

        mock_chunker_instance = MagicMock()
        mock_chunker_instance.chunk_content.side_effect = \
            lambda c: (call_order.append("chunk"), mock_chunks)[1]

        monkeypatch.setattr(
            "backend.rag.chroma.chroma_store.EpubExtractor",
            lambda path: mock_extractor_instance,
        )
        monkeypatch.setattr(
            "backend.rag.chroma.chroma_store.TextCleaner",
            lambda cfg: mock_cleaner_instance,
        )
        monkeypatch.setattr(
            "backend.rag.chroma.chroma_store.TextChunker",
            lambda cfg: mock_chunker_instance,
        )

        epub_path = tmp_path / "fake.epub"
        epub_path.write_bytes(b"fake epub content")

        result = store.ingest(epub_path)

        assert call_order == ["extract", "clean", "chunk"]
        assert isinstance(result, IngestResult)
        assert result.total_chunks == 3

    def test_returns_ingest_result(self, store, tmp_path, monkeypatch):
        mock_chunks = [make_chunk("content", chunk_index=0)]
        mock_content = MagicMock(sections=[])
        mock_content.metadata = {"title": "Test Book", "author": "Test Author"}
        monkeypatch.setattr(
            "backend.rag.chroma.chroma_store.EpubExtractor",
            lambda path: MagicMock(extract=lambda: mock_content),
        )
        monkeypatch.setattr(
            "backend.rag.chroma.chroma_store.TextCleaner",
            lambda cfg: MagicMock(clean_content=lambda c: MagicMock(sections=[])),
        )
        monkeypatch.setattr(
            "backend.rag.chroma.chroma_store.TextChunker",
            lambda cfg: MagicMock(chunk_content=lambda c: mock_chunks),
        )
        epub_path = tmp_path / "fake.epub"
        epub_path.write_bytes(b"fake epub content")
        result = store.ingest(epub_path)
        assert isinstance(result, IngestResult)


# ---------------------------------------------------------------------------
# _resolve_db
# ---------------------------------------------------------------------------

class TestResolveDb:

    def test_same_collection_returns_default_db(self, store, mock_db):
        resolved = store._resolve_db("test_col")
        assert resolved is mock_db

    def test_none_returns_default_db(self, store, mock_db):
        resolved = store._resolve_db(None)
        assert resolved is mock_db

    def test_different_collection_creates_new_db(
        self, store, tmp_path, monkeypatch, mock_db
    ):
        new_db = MagicMock()
        new_db._collection = MagicMock()
        new_db._collection.name = "other_col"

        original_db = mock_db

        def fake_chroma(**kwargs):
            return new_db if kwargs.get("collection_name") == "other_col" else original_db

        monkeypatch.setattr("backend.rag.chroma.chroma_store.Chroma", fake_chroma)
        resolved = store._resolve_db("other_col")
        assert resolved is new_db
