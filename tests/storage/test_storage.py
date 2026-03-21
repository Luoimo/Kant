import pytest
from pathlib import Path
from backend.storage.note_storage import LocalNoteStorage
from backend.storage.plan_storage import LocalPlanStorage


class TestLocalNoteStorage:
    def test_save_and_load(self, tmp_path):
        storage = LocalNoteStorage(root=tmp_path)
        path = storage.save("# Hello", "note_001")
        assert storage.load(path) == "# Hello"

    def test_list_empty(self, tmp_path):
        storage = LocalNoteStorage(root=tmp_path)
        assert storage.list() == []

    def test_list_after_saves(self, tmp_path):
        storage = LocalNoteStorage(root=tmp_path)
        storage.save("A", "note_001")
        storage.save("B", "note_002")
        items = storage.list()
        assert len(items) == 2

    def test_list_with_prefix(self, tmp_path):
        storage = LocalNoteStorage(root=tmp_path)
        storage.save("A", "note_001")
        storage.save("B", "other_001")
        items = storage.list(prefix="note_")
        assert len(items) == 1

    def test_delete(self, tmp_path):
        storage = LocalNoteStorage(root=tmp_path)
        path = storage.save("content", "note_del")
        storage.delete(path)
        assert storage.list() == []

    def test_delete_nonexistent_no_error(self, tmp_path):
        storage = LocalNoteStorage(root=tmp_path)
        storage.delete(str(tmp_path / "nonexistent.md"))  # should not raise

    def test_root_created_if_not_exists(self, tmp_path):
        new_dir = tmp_path / "nested" / "notes"
        storage = LocalNoteStorage(root=new_dir)
        assert new_dir.exists()

    def test_save_returns_string_path(self, tmp_path):
        storage = LocalNoteStorage(root=tmp_path)
        path = storage.save("content", "note_abc")
        assert isinstance(path, str)
        assert path.endswith("note_abc.md")

    def test_load_missing_path_raises(self, tmp_path):
        storage = LocalNoteStorage(root=tmp_path)
        with pytest.raises(FileNotFoundError):
            storage.load(str(tmp_path / "nonexistent.md"))


class TestLocalPlanStorage:
    def test_save_and_load(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        path = storage.save("## Plan", "plan_001")
        assert storage.load(path) == "## Plan"

    def test_list_empty(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        assert storage.list() == []

    def test_list_after_saves(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        storage.save("A", "plan_001")
        storage.save("B", "plan_002")
        items = storage.list()
        assert len(items) == 2

    def test_list_with_prefix(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        storage.save("A", "plan_001")
        storage.save("B", "other_001")
        items = storage.list(prefix="plan_")
        assert len(items) == 1

    def test_delete(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        path = storage.save("content", "plan_del")
        storage.delete(path)
        assert storage.list() == []

    def test_delete_nonexistent_no_error(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        storage.delete(str(tmp_path / "nonexistent.md"))  # should not raise

    def test_root_created_if_not_exists(self, tmp_path):
        new_dir = tmp_path / "nested" / "plans"
        storage = LocalPlanStorage(root=new_dir)
        assert new_dir.exists()

    def test_save_returns_string_path(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        path = storage.save("content", "plan_abc")
        assert isinstance(path, str)
        assert path.endswith("plan_abc.md")

    def test_load_missing_path_raises(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        with pytest.raises(FileNotFoundError):
            storage.load(str(tmp_path / "nonexistent.md"))
