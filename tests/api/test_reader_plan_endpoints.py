"""Tests for Reader Mode plan endpoints."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app

_BOOK_ID = "a1b2c3d4-0000-5000-8000-000000000001"
_BOOK_TITLE = "纯粹理性批判"
_BOOK_ENTRY = {
    "book_id": _BOOK_ID,
    "source": "data/books/kant.epub",
    "book_title": _BOOK_TITLE,
    "author": "康德",
}

_SAMPLE_PLAN = """\
# 《纯粹理性批判》阅读计划

**生成时间：** 2026-03-22
**阅读目标：** 通读全书

## 章节进度

- [ ] 先验感性论（约45分钟）
- [ ] 先验分析论（约90分钟）

## 建议日程

每天阅读一小时，预计两周完成。
"""


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_generator():
    gen = MagicMock()
    gen.generate.return_value = _SAMPLE_PLAN
    return gen


def _mock_store():
    """Return a ChromaStore mock whose resolve_book_by_id returns _BOOK_ENTRY."""
    store = MagicMock()
    store.resolve_book_by_id.return_value = _BOOK_ENTRY
    return store


class TestReaderInitEndpoint:
    def test_init_creates_plan(self, client, mock_generator):
        with patch("backend.api.reader.ChromaStore", return_value=_mock_store()), \
             patch("backend.api.reader.PlanGenerator", return_value=mock_generator):
            resp = client.post(
                f"/reader/{_BOOK_ID}/init",
                json={"reading_goal": "通读"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["book_title"] == _BOOK_TITLE
        assert data["book_id"] == _BOOK_ID
        assert _BOOK_TITLE in data["plan"]

    def test_init_reading_goal_optional(self, client, mock_generator):
        with patch("backend.api.reader.ChromaStore", return_value=_mock_store()), \
             patch("backend.api.reader.PlanGenerator", return_value=mock_generator):
            resp = client.post(f"/reader/{_BOOK_ID}/init", json={})
        assert resp.status_code == 200

    def test_init_404_when_book_not_found(self, client):
        store = MagicMock()
        store.resolve_book_by_id.return_value = None
        with patch("backend.api.reader.ChromaStore", return_value=store):
            resp = client.post("/reader/unknown-uuid/init", json={})
        assert resp.status_code == 404


class TestReaderGetPlanEndpoint:
    def test_get_plan_returns_content(self, client, tmp_path):
        plan_file = tmp_path / f"{_BOOK_TITLE}.md"
        plan_file.write_text(_SAMPLE_PLAN, encoding="utf-8")
        with patch("backend.api.reader.ChromaStore", return_value=_mock_store()), \
             patch("backend.api.reader.get_settings") as mock_settings:
            mock_settings.return_value.plan_storage_dir = str(tmp_path)
            resp = client.get(f"/reader/{_BOOK_ID}/plan")
        assert resp.status_code == 200
        data = resp.json()
        assert data["book_title"] == _BOOK_TITLE
        assert data["book_id"] == _BOOK_ID

    def test_get_plan_returns_empty_when_not_exists(self, client, tmp_path):
        with patch("backend.api.reader.ChromaStore", return_value=_mock_store()), \
             patch("backend.api.reader.get_settings") as mock_settings:
            mock_settings.return_value.plan_storage_dir = str(tmp_path)
            resp = client.get(f"/reader/{_BOOK_ID}/plan")
        assert resp.status_code == 200
        assert resp.json()["plan"] == ""


class TestReaderProgressEndpoint:
    def test_progress_marks_chapter_done(self, client, tmp_path):
        plan_file = tmp_path / f"{_BOOK_TITLE}.md"
        plan_file.write_text(_SAMPLE_PLAN, encoding="utf-8")
        with patch("backend.api.reader.ChromaStore", return_value=_mock_store()), \
             patch("backend.api.reader.get_settings") as mock_settings:
            mock_settings.return_value.plan_storage_dir = str(tmp_path)
            resp = client.post(
                f"/reader/{_BOOK_ID}/progress",
                json={"chapter": "先验感性论"},
            )
        assert resp.status_code == 200
        updated = plan_file.read_text(encoding="utf-8")
        assert "- [x] 先验感性论" in updated
        assert "- [ ] 先验分析论" in updated

    def test_progress_404_when_no_plan(self, client, tmp_path):
        with patch("backend.api.reader.ChromaStore", return_value=_mock_store()), \
             patch("backend.api.reader.get_settings") as mock_settings:
            mock_settings.return_value.plan_storage_dir = str(tmp_path)
            resp = client.post(
                f"/reader/{_BOOK_ID}/progress",
                json={"chapter": "第1章"},
            )
        assert resp.status_code == 404
