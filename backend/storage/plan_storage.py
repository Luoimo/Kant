from __future__ import annotations
import re as _re
from typing import Protocol, runtime_checkable

from backend.storage.note_storage import _LocalMarkdownStorage


def safe_plan_name(book_title: str) -> str:
    """Sanitise a book title into a safe filename stem (no extension).
    Single source of truth — import this everywhere instead of duplicating the regex.
    """
    safe = _re.sub(r'[<>:"/\\|?*《》【】\r\n]', "_", book_title).strip("_. ")
    return safe or "unknown"


@runtime_checkable
class PlanStorage(Protocol):
    def save(self, content: str, plan_id: str) -> str | None: ...
    def load(self, storage_path: str) -> str: ...
    def list(self, prefix: str = "") -> list[str]: ...
    def delete(self, storage_path: str) -> None: ...
    def update(self, storage_path: str, content: str) -> None: ...
    def search(self, query: str, top_k: int = 5) -> list[tuple[str, str]]: ...
    def find_by_book(self, book_title: str) -> str | None: ...


class LocalPlanStorage(_LocalMarkdownStorage):
    """Stores plans as .md files under a configurable root directory."""

    def save(self, content: str, plan_id: str) -> str:
        return self._save(content, plan_id)

    def find_by_book(self, book_title: str) -> str | None:
        """Return storage_path if a plan for this book exists, else None."""
        path = self.root / f"{safe_plan_name(book_title)}.md"
        return str(path) if path.exists() else None


__all__ = ["PlanStorage", "LocalPlanStorage", "safe_plan_name"]
