from __future__ import annotations
from typing import Protocol, runtime_checkable

from backend.storage.note_storage import _LocalMarkdownStorage


@runtime_checkable
class PlanStorage(Protocol):
    def save(self, content: str, plan_id: str) -> str | None: ...
    def load(self, storage_path: str) -> str: ...
    def list(self, prefix: str = "") -> list[str]: ...
    def delete(self, storage_path: str) -> None: ...
    def update(self, storage_path: str, content: str) -> None: ...


class LocalPlanStorage(_LocalMarkdownStorage):
    """Stores plans as .md files under a configurable root directory.

    File lookup (which plan belongs to which book) is handled by PlanCatalog
    (SQLite). This class only performs file I/O.
    """

    def save(self, content: str, plan_id: str) -> str:
        return self._save(content, plan_id)


__all__ = ["PlanStorage", "LocalPlanStorage"]
