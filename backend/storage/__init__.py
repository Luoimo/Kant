from pathlib import Path

from .note_storage import NoteStorage, LocalNoteStorage
from .plan_storage import PlanStorage, LocalPlanStorage


def make_plan_storage(settings) -> PlanStorage:
    """Return a LocalPlanStorage rooted at the configured plan_storage_dir."""
    return LocalPlanStorage(Path(settings.plan_storage_dir))


__all__ = [
    "NoteStorage", "LocalNoteStorage",
    "PlanStorage", "LocalPlanStorage",
    "make_plan_storage",
]
