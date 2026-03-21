from pathlib import Path

from .note_storage import NoteStorage, LocalNoteStorage
from .plan_storage import PlanStorage, LocalPlanStorage


def make_note_storage(settings) -> NoteStorage:
    """Return NotionNoteStorage when NOTION_API_KEY is configured, else LocalNoteStorage."""
    if getattr(settings, "notion_api_key", ""):
        from .notion_storage import NotionNoteStorage
        return NotionNoteStorage(settings)
    return LocalNoteStorage(Path(settings.note_storage_dir))


def make_plan_storage(settings) -> PlanStorage:
    """Return NotionPlanStorage when NOTION_API_KEY is configured, else LocalPlanStorage."""
    if getattr(settings, "notion_api_key", ""):
        from .notion_storage import NotionPlanStorage
        return NotionPlanStorage(settings)
    return LocalPlanStorage(Path(settings.plan_storage_dir))


__all__ = [
    "NoteStorage", "LocalNoteStorage",
    "PlanStorage", "LocalPlanStorage",
    "make_note_storage", "make_plan_storage",
]
