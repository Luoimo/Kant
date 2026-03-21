"""Notion-backed storage implementations for notes and plans."""
from __future__ import annotations

import sys
from typing import Any


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


__all__ = ["_markdown_to_blocks", "_make_raw_code_block"]
