"""Tests for Reader Mode plan endpoints."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.chat import app

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


class TestReaderInitEndpoint:
    def test_init_creates_plan(self, client, mock_generator):
        with patch("backend.api.chat.PlanGenerator", return_value=mock_generator):
            resp = client.post(
                "/reader/纯粹理性批判/init",
                json={"book_source": "kant.epub", "reading_goal": "通读"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["book_title"] == "纯粹理性批判"
        assert "plan" in data
        assert "纯粹理性批判" in data["plan"]

    def test_init_reading_goal_optional(self, client, mock_generator):
        with patch("backend.api.chat.PlanGenerator", return_value=mock_generator):
            resp = client.post(
                "/reader/纯粹理性批判/init",
                json={"book_source": "kant.epub"},
            )
        assert resp.status_code == 200


class TestReaderGetPlanEndpoint:
    def test_get_plan_returns_content(self, client, tmp_path):
        plan_file = tmp_path / "纯粹理性批判.md"
        plan_file.write_text(_SAMPLE_PLAN, encoding="utf-8")
        with patch("backend.api.chat._plan_storage_dir", return_value=tmp_path):
            resp = client.get("/reader/纯粹理性批判/plan")
        assert resp.status_code == 200
        assert resp.json()["book_title"] == "纯粹理性批判"

    def test_get_plan_returns_empty_when_not_exists(self, client, tmp_path):
        with patch("backend.api.chat._plan_storage_dir", return_value=tmp_path):
            resp = client.get("/reader/不存在的书/plan")
        assert resp.status_code == 200
        assert resp.json()["plan"] == ""


class TestReaderProgressEndpoint:
    def test_progress_marks_chapter_done(self, client, tmp_path):
        plan_file = tmp_path / "纯粹理性批判.md"
        plan_file.write_text(_SAMPLE_PLAN, encoding="utf-8")
        with patch("backend.api.chat._plan_storage_dir", return_value=tmp_path):
            resp = client.post(
                "/reader/纯粹理性批判/progress",
                json={"chapter": "先验感性论"},
            )
        assert resp.status_code == 200
        updated = plan_file.read_text(encoding="utf-8")
        assert "- [x] 先验感性论" in updated
        assert "- [ ] 先验分析论" in updated

    def test_progress_404_when_no_plan(self, client, tmp_path):
        with patch("backend.api.chat._plan_storage_dir", return_value=tmp_path):
            resp = client.post(
                "/reader/不存在的书/progress",
                json={"chapter": "第1章"},
            )
        assert resp.status_code == 404
