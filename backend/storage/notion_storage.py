"""Notion-backed storage implementations for notes and plans."""
from __future__ import annotations

import sys
from typing import Any

try:
    from notion_client import Client
except ImportError:
    Client = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Markdown → Notion blocks (minimal custom converter)
# ---------------------------------------------------------------------------

def _rich_text(content: str) -> list[dict]:
    return [{"type": "text", "text": {"content": content}}]


def _markdown_to_blocks(text: str) -> list[dict]:
    """Convert Markdown text to a list of Notion block objects.

    Handles: H1/H2/H3, bullet lists (- and *), paragraphs.
    Blank lines are skipped. Bold markers (**) are left as plain text —
    round-trip fidelity relies on the raw code block, not rendered blocks.
    """
    blocks: list[dict] = []
    for line in text.splitlines():
        stripped = line.rstrip()
        if not stripped:
            continue
        if stripped.startswith("### "):
            blocks.append({"type": "heading_3", "heading_3": {"rich_text": _rich_text(stripped[4:])}})
        elif stripped.startswith("## "):
            blocks.append({"type": "heading_2", "heading_2": {"rich_text": _rich_text(stripped[3:])}})
        elif stripped.startswith("# "):
            blocks.append({"type": "heading_1", "heading_1": {"rich_text": _rich_text(stripped[2:])}})
        elif stripped.startswith("- ") or stripped.startswith("* "):
            blocks.append({"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": _rich_text(stripped[2:])}})
        else:
            blocks.append({"type": "paragraph", "paragraph": {"rich_text": _rich_text(stripped)}})
    return blocks


def _make_raw_code_block(raw_markdown: str) -> dict:
    """Return a Notion code block containing the raw Markdown string.

    This block is appended as the last block on every page, enabling
    lossless load() that doesn't depend on rendered block conversion.
    """
    return {
        "type": "code",
        "code": {
            "rich_text": _rich_text(raw_markdown),
            "language": "markdown",
        },
    }


# ---------------------------------------------------------------------------
# Helper utilities for NotionNoteStorage / NotionPlanStorage
# ---------------------------------------------------------------------------

def _fetch_all_blocks(client: Any, page_id: str) -> list[dict]:
    """Fetch all blocks from a Notion page, following pagination cursors."""
    results: list[dict] = []
    response = client.blocks.children.list(block_id=page_id)
    results.extend(response["results"])
    while response.get("has_more"):
        response = client.blocks.children.list(
            block_id=page_id, start_cursor=response["next_cursor"]
        )
        results.extend(response["results"])
    return results


def _extract_raw_markdown(blocks: list[dict]) -> str | None:
    """Find the last code block and extract its plain_text (raw Markdown)."""
    for block in reversed(blocks):
        if block.get("type") == "code":
            rich_text = block["code"].get("rich_text", [])
            return "".join(rt.get("plain_text", "") for rt in rich_text)
    return None


def _blocks_to_plain_text(blocks: list[dict]) -> str:
    """Fallback: concatenate plain_text from all blocks (degraded quality)."""
    parts: list[str] = []
    for block in blocks:
        btype = block.get("type", "")
        content = block.get(btype, {}).get("rich_text", [])
        text = "".join(rt.get("plain_text", "") for rt in content)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _append_blocks_in_batches(client: Any, page_id: str, blocks: list[dict]) -> None:
    """Append blocks to a Notion page in batches of 100 (API limit)."""
    for i in range(0, len(blocks), 100):
        client.blocks.children.append(block_id=page_id, children=blocks[i:i + 100])


# ---------------------------------------------------------------------------
# Storage classes
# ---------------------------------------------------------------------------

class NotionNoteStorage:
    """Stores notes as pages in a Notion Database."""

    def __init__(self, settings: Any) -> None:
        if not settings.notion_notes_db_id:
            raise ValueError("NOTION_NOTES_DB_ID must be set to use Notion storage")
        if Client is None:
            raise ImportError("notion-client is not installed. Run: pip install notion-client")
        self._client = Client(auth=settings.notion_api_key)
        self._db_id = settings.notion_notes_db_id

    def save(self, content: str, note_id: str) -> str | None:
        try:
            rendered = _markdown_to_blocks(content)
            raw_block = _make_raw_code_block(content)
            all_blocks = rendered + [raw_block]

            page = self._client.pages.create(
                parent={"database_id": self._db_id},
                properties={
                    "Title": {"title": [{"text": {"content": note_id}}]},
                },
                children=all_blocks[:100],
            )
            page_id: str = page["id"]

            if len(all_blocks) > 100:
                _append_blocks_in_batches(self._client, page_id, all_blocks[100:])

            return page_id
        except Exception as e:
            print(f"[NotionNoteStorage] save failed: {e}", file=sys.stdout)
            return None

    def load(self, storage_path: str) -> str:
        blocks = _fetch_all_blocks(self._client, storage_path)
        raw = _extract_raw_markdown(blocks)
        if raw is not None:
            return raw
        return _blocks_to_plain_text(blocks)

    def update(self, storage_path: str, content: str) -> None:
        blocks = _fetch_all_blocks(self._client, storage_path)
        for block in blocks:
            self._client.blocks.delete(block_id=block["id"])
        new_blocks = _markdown_to_blocks(content) + [_make_raw_code_block(content)]
        _append_blocks_in_batches(self._client, storage_path, new_blocks)

    def list(self, prefix: str = "") -> list[str]:
        filter_: dict = {}
        if prefix:
            filter_ = {"property": "Title", "title": {"starts_with": prefix}}
        response = self._client.databases.query(database_id=self._db_id, filter=filter_)
        page_ids = [p["id"] for p in response["results"]]
        while response.get("has_more"):
            response = self._client.databases.query(
                database_id=self._db_id,
                filter=filter_,
                start_cursor=response["next_cursor"],
            )
            page_ids.extend(p["id"] for p in response["results"])
        return page_ids

    def delete(self, storage_path: str) -> None:
        self._client.pages.update(page_id=storage_path, archived=True)


class NotionPlanStorage:
    """Stores reading plans as pages in a Notion Database. Identical pattern to NotionNoteStorage."""

    def __init__(self, settings: Any) -> None:
        if not settings.notion_plans_db_id:
            raise ValueError("NOTION_PLANS_DB_ID must be set to use Notion storage")
        if Client is None:
            raise ImportError("notion-client is not installed. Run: pip install notion-client")
        self._client = Client(auth=settings.notion_api_key)
        self._db_id = settings.notion_plans_db_id

    def save(self, content: str, plan_id: str) -> str | None:
        try:
            rendered = _markdown_to_blocks(content)
            raw_block = _make_raw_code_block(content)
            all_blocks = rendered + [raw_block]

            page = self._client.pages.create(
                parent={"database_id": self._db_id},
                properties={
                    "Title": {"title": [{"text": {"content": plan_id}}]},
                },
                children=all_blocks[:100],
            )
            page_id: str = page["id"]

            if len(all_blocks) > 100:
                _append_blocks_in_batches(self._client, page_id, all_blocks[100:])

            return page_id
        except Exception as e:
            print(f"[NotionPlanStorage] save failed: {e}", file=sys.stdout)
            return None

    def load(self, storage_path: str) -> str:
        blocks = _fetch_all_blocks(self._client, storage_path)
        raw = _extract_raw_markdown(blocks)
        if raw is not None:
            return raw
        return _blocks_to_plain_text(blocks)

    def update(self, storage_path: str, content: str) -> None:
        blocks = _fetch_all_blocks(self._client, storage_path)
        for block in blocks:
            self._client.blocks.delete(block_id=block["id"])
        new_blocks = _markdown_to_blocks(content) + [_make_raw_code_block(content)]
        _append_blocks_in_batches(self._client, storage_path, new_blocks)

    def list(self, prefix: str = "") -> list[str]:
        filter_: dict = {}
        if prefix:
            filter_ = {"property": "Title", "title": {"starts_with": prefix}}
        response = self._client.databases.query(database_id=self._db_id, filter=filter_)
        page_ids = [p["id"] for p in response["results"]]
        while response.get("has_more"):
            response = self._client.databases.query(
                database_id=self._db_id,
                filter=filter_,
                start_cursor=response["next_cursor"],
            )
            page_ids.extend(p["id"] for p in response["results"])
        return page_ids

    def delete(self, storage_path: str) -> None:
        self._client.pages.update(page_id=storage_path, archived=True)


__all__ = ["NotionNoteStorage", "NotionPlanStorage"]

