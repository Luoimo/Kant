from __future__ import annotations
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class NoteStorage(Protocol):
    def save(self, content: str, note_id: str) -> str: ...   # returns storage_path
    def load(self, storage_path: str) -> str: ...
    def list(self, prefix: str = "") -> list[str]: ...
    def delete(self, storage_path: str) -> None: ...


class _LocalMarkdownStorage:
    """Base: stores content as .md files under a root directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _save(self, content: str, file_id: str) -> str:
        path = self.root / f"{file_id}.md"
        path.write_text(content, encoding="utf-8")
        return str(path)

    def load(self, storage_path: str) -> str:
        return Path(storage_path).read_text(encoding="utf-8")

    def list(self, prefix: str = "") -> list[str]:
        return sorted(str(p) for p in self.root.glob(f"{prefix}*.md"))

    def delete(self, storage_path: str) -> None:
        Path(storage_path).unlink(missing_ok=True)


class LocalNoteStorage(_LocalMarkdownStorage):
    """Stores notes as .md files under a configurable root directory."""

    def save(self, content: str, note_id: str) -> str:
        return self._save(content, note_id)


__all__ = ["NoteStorage", "LocalNoteStorage"]
