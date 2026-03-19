from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class Citation:
    source: str
    book_title: str | None = None
    author: str | None = None
    section_indices: list[int] | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    snippet: str | None = None
    chapter_title: str | None = None  # 书本章标题（来自 TOC）
    section_title: str | None = None  # 书本节标题（来自 TOC）


def _parse_section_indices(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[int] = []
        for x in value:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        out = []
        for p in parts:
            try:
                out.append(int(p))
            except Exception:
                continue
        return out
    try:
        return [int(value)]  # type: ignore[arg-type]
    except Exception:
        return []


def build_citations(docs: list[Document], *, snippet_chars: int = 240) -> list[Citation]:
    citations: list[Citation] = []
    for d in docs:
        m = d.metadata or {}
        source = str(m.get("source") or "")
        if not source:
            continue

        text = (d.page_content or "").strip()
        snippet = text[:snippet_chars] + ("…" if len(text) > snippet_chars else "")

        citations.append(
            Citation(
                source=source,
                book_title=m.get("book_title"),
                author=m.get("author"),
                section_indices=_parse_section_indices(m.get("section_indices")),
                chunk_id=m.get("chunk_id"),
                chunk_index=m.get("chunk_index"),
                snippet=snippet,
                chapter_title=m.get("chapter_title") or None,
                section_title=m.get("section_title") or None,
            )
        )
    return citations
