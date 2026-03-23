import pytest
from pathlib import Path
from backend.storage.note_storage import LocalNoteStorage
from backend.storage.plan_storage import LocalPlanStorage, safe_plan_name


class TestSafePlanName:
    def test_strips_chinese_brackets(self):
        assert safe_plan_name("《纯粹理性批判》") == "纯粹理性批判"

    def test_strips_angle_brackets(self):
        assert safe_plan_name("<test>") == "test"

    def test_fallback_for_empty(self):
        assert safe_plan_name("") == "unknown"


class TestLocalPlanStorageFindByBook:
    def test_find_by_book_returns_path_when_exists(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        storage.save("plan content", safe_plan_name("纯粹理性批判"))
        path = storage.find_by_book("纯粹理性批判")
        assert path is not None
        assert path.endswith(".md")

    def test_find_by_book_returns_none_when_missing(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        assert storage.find_by_book("不存在的书") is None

    def test_find_by_book_sanitizes_title(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        storage.save("content", safe_plan_name("《纯粹理性批判》"))
        path = storage.find_by_book("《纯粹理性批判》")
        assert path is not None


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

    def test_update_overwrites_content(self, tmp_path):
        storage = LocalNoteStorage(root=tmp_path)
        path = storage.save("# Original", "note_001")
        storage.update(path, "# Updated")
        assert storage.load(path) == "# Updated"

    def test_save_returns_str_or_none(self, tmp_path):
        storage = LocalNoteStorage(root=tmp_path)
        result = storage.save("content", "note_x")
        assert result is None or isinstance(result, str)


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

    def test_update_overwrites_content(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        path = storage.save("# Original", "plan_001")
        storage.update(path, "# Updated")
        assert storage.load(path) == "# Updated"

    def test_save_returns_str_or_none(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        result = storage.save("content", "plan_x")
        assert result is None or isinstance(result, str)

    def test_mark_chapter_done_toggles_checkbox(self, tmp_path):
        content = "## 章节进度\n\n- [ ] 先验感性论（约45分钟）\n- [ ] 先验分析论（约90分钟）\n"
        storage = LocalPlanStorage(root=tmp_path)
        storage.save(content, safe_plan_name("纯粹理性批判"))
        result = storage.mark_chapter_done("纯粹理性批判", "先验感性论")
        assert result is True
        updated = (tmp_path / "纯粹理性批判.md").read_text(encoding="utf-8")
        assert "- [x] 先验感性论" in updated
        assert "- [ ] 先验分析论" in updated

    def test_mark_chapter_done_returns_false_when_not_found(self, tmp_path):
        content = "## 章节进度\n\n- [ ] 先验感性论（约45分钟）\n"
        storage = LocalPlanStorage(root=tmp_path)
        storage.save(content, safe_plan_name("纯粹理性批判"))
        result = storage.mark_chapter_done("纯粹理性批判", "不存在的章节")
        assert result is False

    def test_mark_chapter_done_returns_false_when_no_plan(self, tmp_path):
        storage = LocalPlanStorage(root=tmp_path)
        result = storage.mark_chapter_done("不存在的书", "第1章")
        assert result is False
