from __future__ import annotations

from storage.note_storage import _LocalMarkdownStorage


class LocalPlanStorage(_LocalMarkdownStorage):
    """Stores plans as .md files under a configurable root directory.

    File lookup (which plan belongs to which book) is handled by PlanCatalog
    (SQLite). This class only performs file I/O.
    """

    def save(self, content: str, plan_id: str) -> str:
        return self._save(content, plan_id)


__all__ = ["LocalPlanStorage"]
