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
