"""NoteService — storage-level CRUD for notes.

No LLM. No agent logic. Handles:
- append_manual: user-submitted handwritten notes
- get_note_content: read note file
- list_books: list books with notes (via SQL catalog)
- get_timeline: structured metadata for frontend timeline
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from storage.book_catalog import get_note_catalog, get_book_catalog
from utils.text import safe_id


class NoteService:
    def __init__(
        self,
        *,
        notes_dir: Path,
        note_vector_store=None,
    ) -> None:
        self._notes_dir = Path(notes_dir)
        self._notes_dir.mkdir(parents=True, exist_ok=True)
        self._note_vector_store = note_vector_store

    def _note_path(self, book_id: str) -> Path:
        """Return note file path for a book_id, creating catalog entry if needed."""
        record = get_note_catalog().get_by_book_id(book_id)
        if record:
            return Path(record["file_path"])
        path = self._notes_dir / f"{book_id}.md"
        get_note_catalog().upsert(book_id=book_id, file_path=str(path))
        return path

    def append_manual(self, content: str, book_id: str) -> None:
        """追加用户手写笔记到指定书籍的笔记文件。"""
        now = datetime.now(tz=timezone.utc)
        display_date = now.strftime("%Y-%m-%d %H:%M")

        md = (
            f"\n---\n"
            f"## {display_date} · 手记\n\n"
            f"{content.strip()}\n"
        )
        path = self._note_path(book_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(md)
        get_note_catalog().touch(book_id)

        if self._note_vector_store:
            book = get_book_catalog().get_by_id(book_id)
            book_title = book["title"] if book else book_id
            entry_id = f"manual_{safe_id(book_title)}_{int(now.timestamp())}"
            self._note_vector_store.add_entry(
                entry_id=entry_id,
                content=content,
                metadata={
                    "book_title": book_title,
                    "date": now.isoformat(),
                    "question_summary": content[:60],
                    "concepts": "[]",
                    "entry_type": "manual",
                },
            )

    def get_note_content(self, book_id: str) -> str:
        """返回笔记文件全文，供前端编辑器展示。"""
        record = get_note_catalog().get_by_book_id(book_id)
        if not record:
            return ""
        path = Path(record["file_path"])
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def list_books(self) -> list[dict]:
        """返回已有笔记的书籍列表，含 book_id 和 title。"""
        book_catalog = get_book_catalog()
        result = []
        for note in get_note_catalog().get_all():
            book = book_catalog.get_by_id(note["book_id"])
            result.append({
                "book_id": note["book_id"],
                "title": book["title"] if book else note["book_id"],
                "updated_at": note["updated_at"],
            })
        return result

    def get_timeline(self, book_id: str | None = None) -> dict:
        """返回结构化元数据供前端时间轴可视化。"""
        if self._note_vector_store:
            # vector store still works with book_title; look it up
            book_title = None
            if book_id:
                book = get_book_catalog().get_by_id(book_id)
                book_title = book["title"] if book else None
            return self._note_vector_store.get_timeline(book_title)
        return self._timeline_from_files(book_id)

    def _timeline_from_files(self, book_id: str | None) -> dict:
        """Fallback: parse .md headers directly."""
        notes = (
            [get_note_catalog().get_by_book_id(book_id)]
            if book_id
            else get_note_catalog().get_all()
        )
        notes = [n for n in notes if n]
        book_catalog = get_book_catalog()
        entries = []
        books = []
        for note in notes:
            bid = note["book_id"]
            book = book_catalog.get_by_id(bid)
            title = book["title"] if book else bid
            books.append(title)
            path = Path(note["file_path"])
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("## ") and " · " in line:
                    parts = line[3:].split(" · ", 1)
                    entries.append({
                        "date": parts[0].strip(),
                        "book_title": title,
                        "question_summary": parts[1].strip() if len(parts) > 1 else "",
                        "concepts": [],
                        "entry_type": "qa",
                    })
        return {"entries": entries, "books": books, "concept_frequency": {}}


__all__ = ["NoteService"]
