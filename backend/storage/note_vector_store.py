"""ChromaDB-backed vector store for note entries (kant_notes collection).

Separate from the main book collection (kant_library).
Used for semantic search and cross-book association.
"""
from __future__ import annotations

import json
import sys
from typing import Any

import chromadb

from backend.config import get_settings
from backend.llm.openai_client import get_embeddings


class NoteVectorStore:
    COLLECTION_NAME = "kant_notes"

    def __init__(self, settings=None) -> None:
        s = settings or get_settings()
        self._embeddings = get_embeddings()

        if getattr(s, "chroma_api_key", ""):
            self._client = chromadb.CloudClient(
                tenant=s.chroma_tenant,
                database=s.chroma_database,
                api_key=s.chroma_api_key,
            )
        else:
            import os
            os.makedirs(s.chroma_persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=s.chroma_persist_dir)

        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def add_entry(self, entry_id: str, content: str, metadata: dict[str, Any]) -> None:
        """Upsert a note entry (embedding computed here)."""
        embedding = self._embeddings.embed_documents([content])[0]
        self._collection.upsert(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
        )

    # ------------------------------------------------------------------
    # 跨书关联搜索
    # ------------------------------------------------------------------

    def search_similar(
        self,
        text: str,
        exclude_book: str,
        top_k: int = 2,
        max_distance: float = 0.5,   # cosine distance: 0=identical, 2=opposite; <0.5 = quite similar
    ) -> list[dict]:
        """Return semantically similar entries from other books."""
        try:
            total = self._collection.count()
            if total == 0:
                return []

            where: dict | None = {"book_title": {"$ne": exclude_book}} if exclude_book else None

            results = self._collection.query(
                query_embeddings=[self._embeddings.embed_query(text)],
                n_results=min(top_k + 5, total),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"[NoteVectorStore] search_similar failed: {e}", file=sys.stderr)
            return []

        out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist <= max_distance:
                out.append({
                    "content": doc,
                    "book_title": meta.get("book_title", ""),
                    "question_summary": meta.get("question_summary", ""),
                    "date": meta.get("date", "")[:10],
                })
        return out[:top_k]

    # ------------------------------------------------------------------
    # 时间轴数据（供前端可视化）
    # ------------------------------------------------------------------

    def get_timeline(self, book_title: str | None = None) -> dict:
        """Return all note metadata structured for frontend timeline visualization."""
        try:
            where: dict | None = {"book_title": book_title} if book_title else None
            results = self._collection.get(where=where, include=["metadatas"])
        except Exception as e:
            print(f"[NoteVectorStore] get_timeline failed: {e}", file=sys.stderr)
            return {"entries": [], "books": [], "concept_frequency": {}}

        entries = []
        books: set[str] = set()
        concept_freq: dict[str, int] = {}

        for meta in results["metadatas"]:
            b = meta.get("book_title", "")
            books.add(b)
            concepts: list[str] = json.loads(meta.get("concepts", "[]"))
            for c in concepts:
                concept_freq[c] = concept_freq.get(c, 0) + 1
            entries.append({
                "date": meta.get("date", ""),
                "book_title": b,
                "question_summary": meta.get("question_summary", ""),
                "concepts": concepts,
                "entry_type": meta.get("entry_type", "qa"),
            })

        entries.sort(key=lambda x: x["date"])
        return {
            "entries": entries,
            "books": sorted(books),
            "concept_frequency": dict(
                sorted(concept_freq.items(), key=lambda x: -x[1])
            ),
        }


def make_note_vector_store(settings=None) -> NoteVectorStore | None:
    """Factory — returns None gracefully if setup fails."""
    try:
        return NoteVectorStore(settings)
    except Exception as e:
        print(f"[NoteVectorStore] init failed, cross-book association disabled: {e}", file=sys.stderr)
        return None


__all__ = ["NoteVectorStore", "make_note_vector_store"]
