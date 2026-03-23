from __future__ import annotations
from typing import Protocol, runtime_checkable

from backend.storage.note_storage import _LocalMarkdownStorage


@runtime_checkable
class PlanStorage(Protocol):
    def save(self, content: str, plan_id: str) -> str: ...
    def load(self, storage_path: str) -> str: ...
    def list(self, prefix: str = "") -> list[str]: ...
    def delete(self, storage_path: str) -> None: ...


class LocalPlanStorage(_LocalMarkdownStorage):
    """Stores plans as .md files under a configurable root directory."""

    def save(self, content: str, plan_id: str) -> str:
        return self._save(content, plan_id)


__all__ = ["PlanStorage", "LocalPlanStorage"]
