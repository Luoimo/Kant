"""Tests for Notion storage — all Notion API calls are mocked."""
from __future__ import annotations
import pytest
from backend.storage.notion_storage import _markdown_to_blocks


class TestMarkdownToBlocks:
    def test_heading1(self):
        blocks = _markdown_to_blocks("# Title")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "heading_1"
        assert blocks[0]["heading_1"]["rich_text"][0]["text"]["content"] == "Title"

    def test_heading2(self):
        blocks = _markdown_to_blocks("## Section")
        assert blocks[0]["type"] == "heading_2"
        assert blocks[0]["heading_2"]["rich_text"][0]["text"]["content"] == "Section"

    def test_heading3(self):
        blocks = _markdown_to_blocks("### Sub")
        assert blocks[0]["type"] == "heading_3"
        assert blocks[0]["heading_3"]["rich_text"][0]["text"]["content"] == "Sub"

    def test_bullet_dash(self):
        blocks = _markdown_to_blocks("- item one")
        assert blocks[0]["type"] == "bulleted_list_item"
        assert blocks[0]["bulleted_list_item"]["rich_text"][0]["text"]["content"] == "item one"

    def test_bullet_star(self):
        blocks = _markdown_to_blocks("* item two")
        assert blocks[0]["type"] == "bulleted_list_item"
        assert blocks[0]["bulleted_list_item"]["rich_text"][0]["text"]["content"] == "item two"

    def test_paragraph(self):
        blocks = _markdown_to_blocks("plain text here")
        assert blocks[0]["type"] == "paragraph"
        assert blocks[0]["paragraph"]["rich_text"][0]["text"]["content"] == "plain text here"

    def test_blank_lines_skipped(self):
        blocks = _markdown_to_blocks("# H\n\n- item\n\n")
        assert len(blocks) == 2

    def test_mixed_content(self):
        md = "# Title\n## Section\n- bullet\nParagraph text"
        blocks = _markdown_to_blocks(md)
        types = [b["type"] for b in blocks]
        assert types == ["heading_1", "heading_2", "bulleted_list_item", "paragraph"]

    def test_raw_code_block_structure(self):
        """The raw Markdown is always stored as the last code block."""
        from backend.storage.notion_storage import _make_raw_code_block
        block = _make_raw_code_block("# hello")
        assert block["type"] == "code"
        assert block["code"]["rich_text"][0]["text"]["content"] == "# hello"
        assert block["code"]["language"] == "markdown"


from unittest.mock import MagicMock, patch, call
from backend.storage.notion_storage import NotionNoteStorage, NotionPlanStorage


def _make_notion_settings(db_id="notes-db-id"):
    s = MagicMock()
    s.notion_api_key = "secret_test"
    s.notion_notes_db_id = db_id
    s.notion_plans_db_id = "plans-db-id"
    return s


class TestNotionNoteStorage:
    def _make_storage(self, mock_client):
        with patch("backend.storage.notion_storage.Client", return_value=mock_client):
            return NotionNoteStorage(_make_notion_settings())

    def test_save_creates_page_and_returns_page_id(self):
        client = MagicMock()
        client.pages.create.return_value = {"id": "page-uuid-001"}
        storage = self._make_storage(client)

        result = storage.save("# Hello", "note_001")

        assert result == "page-uuid-001"
        client.pages.create.assert_called_once()
        call_kwargs = client.pages.create.call_args[1]
        assert call_kwargs["parent"]["database_id"] == "notes-db-id"
        assert call_kwargs["properties"]["Title"]["title"][0]["text"]["content"] == "note_001"

    def test_save_returns_none_on_api_error(self):
        client = MagicMock()
        client.pages.create.side_effect = Exception("API error")
        storage = self._make_storage(client)

        result = storage.save("# Hello", "note_001")
        assert result is None

    def test_load_extracts_raw_markdown_from_code_block(self):
        client = MagicMock()
        client.blocks.children.list.return_value = {
            "results": [
                {"type": "heading_1", "heading_1": {"rich_text": [{"plain_text": "Title"}]}},
                {
                    "type": "code",
                    "code": {"rich_text": [{"plain_text": "# Title\n\n- bullet"}]},
                },
            ],
            "has_more": False,
        }
        storage = self._make_storage(client)

        result = storage.load("page-uuid-001")
        assert result == "# Title\n\n- bullet"

    def test_load_raises_on_api_error(self):
        client = MagicMock()
        client.blocks.children.list.side_effect = Exception("Not found")
        storage = self._make_storage(client)

        with pytest.raises(Exception, match="Not found"):
            storage.load("page-uuid-001")

    def test_update_deletes_old_blocks_and_appends_new(self):
        client = MagicMock()
        client.blocks.children.list.return_value = {
            "results": [{"id": "block-1"}, {"id": "block-2"}],
            "has_more": False,
        }
        storage = self._make_storage(client)
        storage.update("page-uuid-001", "# New Content")

        assert client.blocks.delete.call_count == 2
        client.blocks.delete.assert_any_call(block_id="block-1")
        client.blocks.delete.assert_any_call(block_id="block-2")
        client.blocks.children.append.assert_called_once()

    def test_list_queries_database_with_title_filter(self):
        client = MagicMock()
        client.databases.query.return_value = {
            "results": [{"id": "page-aaa"}, {"id": "page-bbb"}],
            "has_more": False,
        }
        storage = self._make_storage(client)

        result = storage.list(prefix="note_thread1")
        assert result == ["page-aaa", "page-bbb"]
        call_kwargs = client.databases.query.call_args[1]
        assert call_kwargs["filter"]["title"]["starts_with"] == "note_thread1"

    def test_delete_archives_page(self):
        client = MagicMock()
        storage = self._make_storage(client)
        storage.delete("page-uuid-001")
        client.pages.update.assert_called_once_with(page_id="page-uuid-001", archived=True)

    def test_init_raises_if_db_id_missing(self):
        s = _make_notion_settings(db_id="")
        with patch("backend.storage.notion_storage.Client"):
            with pytest.raises(ValueError, match="NOTION_NOTES_DB_ID"):
                NotionNoteStorage(s)


class TestNotionPlanStorage:
    def _make_storage(self, mock_client):
        with patch("backend.storage.notion_storage.Client", return_value=mock_client):
            return NotionPlanStorage(_make_notion_settings())

    def test_save_creates_page_and_returns_page_id(self):
        client = MagicMock()
        client.pages.create.return_value = {"id": "page-plan-001"}
        storage = self._make_storage(client)
        result = storage.save("## Plan", "plan_001")
        assert result == "page-plan-001"

    def test_init_raises_if_db_id_missing(self):
        s = _make_notion_settings()
        s.notion_plans_db_id = ""
        with patch("backend.storage.notion_storage.Client"):
            with pytest.raises(ValueError, match="NOTION_PLANS_DB_ID"):
                NotionPlanStorage(s)
