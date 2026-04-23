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
    ) -> None:
        self._notes_dir = Path(notes_dir)
        self._notes_dir.mkdir(parents=True, exist_ok=True)

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
        """返回结构化元数据供前端时间轴可视化。解析 .md 文件直接提取"""
        notes = (
            [get_note_catalog().get_by_book_id(book_id)]
            if book_id
            else get_note_catalog().get_all()
        )
        notes = [n for n in notes if n]
        book_catalog = get_book_catalog()
        entries = []
        books = []
        concept_freq: dict[str, int] = {}
        
        import re
        
        for note in notes:
            bid = note["book_id"]
            book = book_catalog.get_by_id(bid)
            title = book["title"] if book else bid
            books.append(title)
            path = Path(note["file_path"])
            if not path.exists():
                continue
            
            content = path.read_text(encoding="utf-8")
            # 找到所有的 `## YYYY-MM-DD HH:MM · Summary`
            # 以及它们之后的所有文本，直到下一个 `## ` 出现
            sections = re.split(r'\n(?=## \d{4}-\d{2}-\d{2} \d{2}:\d{2} \· )', "\n" + content)
            
            for section in sections:
                if not section.strip():
                    continue
                lines = section.strip().splitlines()
                header = lines[0]
                if header.startswith("## ") and " · " in header:
                    parts = header[3:].split(" · ", 1)
                    date_str = parts[0].strip()
                    summary = parts[1].strip() if len(parts) > 1 else ""
                    
                    # 从 section 中提取 #标签 作为 concepts
                    tags = set(re.findall(r'#([^\s#]+)', section))
                    for t in tags:
                        concept_freq[t] = concept_freq.get(t, 0) + 1
                        
                    entries.append({
                        "date": date_str,
                        "book_title": title,
                        "question_summary": summary,
                        "concepts": list(tags),
                        "entry_type": "manual" if summary == "手记" else "qa",
                    })
                    
        entries.sort(key=lambda x: x["date"], reverse=True)
        return {
            "entries": entries, 
            "books": sorted(set(books)), 
            "concept_frequency": dict(sorted(concept_freq.items(), key=lambda x: -x[1]))
        }


__all__ = ["NoteService"]
