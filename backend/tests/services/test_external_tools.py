from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agents.notion_tools as notion
from memory.mem0_store import Mem0Store, USER_MEMORY_EXTRACTION_PROMPT


def page(title: str, page_id: str = "page-1", url: str = "https://notion/page") -> dict:
    return {
        "id": page_id,
        "url": url,
        "properties": {"title": {"title": [{"plain_text": title}]}},
    }


def test_notion_client_cache_and_page_operations(monkeypatch) -> None:
    notion._client_cache.clear()
    client = MagicMock()
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr(notion, "_NotionClient", constructor)
    monkeypatch.setattr(
        notion,
        "get_settings",
        lambda: SimpleNamespace(notion_api_key="token", notion_parent_page_id="parent"),
    )

    assert notion._get_client() is client
    assert notion._get_client() is client
    constructor.assert_called_once_with(auth="token")
    assert notion._get_parent_page_id() == "parent"

    client.search.return_value = {"results": [page("Kant"), page("Other", "page-2")]}
    assert notion._find_book_page("Kant") == "page-1"
    client.search.return_value = {"results": [page("Similar", "fallback")]}
    assert notion._find_book_page("Kant") == "fallback"
    assert notion._find_book_page("") is None
    client.search.side_effect = RuntimeError("offline")
    assert notion._find_book_page("Kant") is None

    client.search.side_effect = None
    client.pages.create.return_value = {"id": "created"}
    assert notion._create_book_page("Kant") == "created"
    client.pages.create.side_effect = RuntimeError("denied")
    assert notion._create_book_page("Kant") is None


@pytest.mark.parametrize(
    "block, expected",
    [
        ({"type": "heading_1", "heading_1": {"rich_text": [{"plain_text": "H1"}]}}, "# H1"),
        ({"type": "heading_2", "heading_2": {"rich_text": [{"plain_text": "H2"}]}}, "## H2"),
        ({"type": "heading_3", "heading_3": {"rich_text": [{"plain_text": "H3"}]}}, "### H3"),
        ({"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "item"}]}}, "- item"),
        ({"type": "numbered_list_item", "numbered_list_item": {"rich_text": [{"plain_text": "item"}]}}, "1. item"),
        ({"type": "quote", "quote": {"rich_text": [{"plain_text": "quote"}]}}, "> quote"),
        ({"type": "code", "code": {"language": "python", "rich_text": [{"plain_text": "x=1"}]}}, "```python\nx=1\n```"),
        ({"type": "divider", "divider": {}}, "---"),
        ({"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "text"}]}}, "text"),
    ],
)
def test_notion_block_to_markdown(block, expected) -> None:
    assert notion._block_to_markdown(block) == expected


def test_notion_read_append_and_markdown_conversion(monkeypatch) -> None:
    client = MagicMock()
    monkeypatch.setattr(notion, "_get_client", lambda: client)
    client.blocks.children.list.side_effect = [
        {
            "results": [
                {"type": "heading_1", "heading_1": {"rich_text": [{"plain_text": "Notes"}]}},
                {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "First"}]}},
            ],
            "has_more": True,
            "next_cursor": "next",
        },
        {
            "results": [{"type": "quote", "quote": {"rich_text": [{"plain_text": "Second"}]}}],
            "has_more": False,
        },
    ]
    assert notion._read_page_markdown("page") == "# Notes\nFirst\n> Second"
    assert client.blocks.children.list.call_args_list[1].kwargs["start_cursor"] == "next"
    assert notion._read_page_markdown("") == ""

    markdown = "# H1\n## H2\n### H3\n- bullet\n* second\n1. numbered\n> quote\n\nparagraph"
    blocks = notion._markdown_to_blocks(markdown)
    assert [item["type"] for item in blocks] == [
        "heading_1", "heading_2", "heading_3", "bulleted_list_item",
        "bulleted_list_item", "numbered_list_item", "quote", "paragraph", "paragraph",
    ]
    assert notion._rich("") == []
    assert len(notion._rich("x" * 3900)) == 3

    monkeypatch.setattr(notion, "_markdown_to_blocks", lambda md: [{"type": "paragraph"}] * 91)
    assert notion._append_markdown("page", "content") is True
    assert client.blocks.children.append.call_count == 2
    monkeypatch.setattr(notion, "_markdown_to_blocks", lambda md: [])
    assert notion._append_markdown("page", "") is False
    assert notion._append_markdown("", "content") is False

    monkeypatch.setattr(notion, "_markdown_to_blocks", lambda md: [{"type": "paragraph"}])
    client.blocks.children.append.side_effect = RuntimeError("denied")
    assert notion._append_markdown("page", "content") is False


def test_notion_langchain_tools_cover_success_and_failure_paths(monkeypatch) -> None:
    monkeypatch.setattr(notion, "_get_client", lambda: None)
    assert "未配置" in notion.read_past_notes.func("Kant")
    assert "未配置" in notion.search_vault_for_concept.func("自由")
    assert "未配置" in notion.append_note_to_workspace.func("Kant", "note")

    client = MagicMock()
    monkeypatch.setattr(notion, "_get_client", lambda: client)
    monkeypatch.setattr(notion, "_find_book_page", lambda title: None)
    assert "暂无" in notion.read_past_notes.func("Kant")
    monkeypatch.setattr(notion, "_find_book_page", lambda title: "page")
    monkeypatch.setattr(notion, "_read_page_markdown", lambda page_id: "saved note")
    assert notion.read_past_notes.func("Kant") == "saved note"

    client.search.return_value = {"results": [page("Freedom"), page("", "p2", "") ]}
    result = notion.search_vault_for_concept.func("freedom")
    assert "Freedom" in result
    assert "(无标题)" in result
    client.search.return_value = {"results": []}
    assert "未在 Notion" in notion.search_vault_for_concept.func("missing")
    client.search.side_effect = RuntimeError("offline")
    assert notion.search_vault_for_concept.func("error") == "Error: offline"

    client.search.side_effect = None
    assert "内容为空" in notion.append_note_to_workspace.func("Kant", " ")
    monkeypatch.setattr(notion, "_find_book_page", lambda title: None)
    monkeypatch.setattr(notion, "_create_book_page", lambda title: None)
    assert "创建 Notion 页面失败" in notion.append_note_to_workspace.func("Kant", "note")
    monkeypatch.setattr(notion, "_create_book_page", lambda title: "created")
    monkeypatch.setattr(notion, "_append_markdown", lambda page_id, content: False)
    assert "追加 Notion 内容失败" in notion.append_note_to_workspace.func("Kant", "note")
    monkeypatch.setattr(notion, "_append_markdown", lambda page_id, content: True)
    assert "已写入" in notion.append_note_to_workspace.func("Kant", "note")


def test_mem0_initialization_cloud_fallback_and_local(monkeypatch) -> None:
    settings = SimpleNamespace(
        chroma_api_key="cloud-key",
        chroma_tenant="tenant",
        chroma_persist_dir="/tmp/chroma",
    )
    monkeypatch.setattr("config.get_settings", lambda: settings)
    memory = MagicMock()
    factory = MagicMock(return_value=memory)
    monkeypatch.setitem(sys.modules, "mem0", SimpleNamespace(Memory=SimpleNamespace(from_config=factory)))
    store = Mem0Store()
    assert store._enabled is True
    cloud_config = factory.call_args.args[0]
    assert cloud_config["vector_store"]["config"]["api_key"] == "cloud-key"
    assert cloud_config["custom_instructions"] == USER_MEMORY_EXTRACTION_PROMPT

    local_memory = MagicMock()
    factory.reset_mock()
    factory.side_effect = [RuntimeError("quota"), local_memory]
    fallback = Mem0Store()
    assert fallback._client is local_memory
    fallback_config = factory.call_args_list[1].args[0]
    assert fallback_config["vector_store"]["config"]["path"] == "/tmp/chroma"
    assert "api_key" not in fallback_config["vector_store"]["config"]

    settings.chroma_api_key = ""
    factory.side_effect = None
    factory.return_value = memory
    local = Mem0Store()
    assert local._enabled is True
    assert factory.call_args.args[0]["vector_store"]["config"]["path"] == "/tmp/chroma"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ([{"memory": "one"}, {"other": "skip"}], ["one"]),
        ({"results": [{"memory": "two"}]}, ["two"]),
        ({"memories": [{"memory": "three"}]}, ["three"]),
        ("unexpected", []),
    ],
)
def test_mem0_search_shapes_and_write_operations(raw, expected) -> None:
    store = object.__new__(Mem0Store)
    store._enabled = True
    store._client = MagicMock()
    store._client.search.return_value = raw

    assert store.search(user_id="u1", query="q", top_k=2) == expected
    store.add_qa(user_id="u1", query="question", answer="answer")
    messages = store._client.add.call_args.kwargs["messages"]
    assert messages == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    store.delete_all(user_id="u1")
    store._client.delete_all.assert_called_once_with(user_id="u1")


def test_mem0_disabled_and_errors_degrade_safely() -> None:
    store = object.__new__(Mem0Store)
    store._enabled = False
    store._client = MagicMock()
    assert store.search(user_id="u", query="q") == []
    assert store.add_qa(user_id="u", query="q", answer="a") is None
    assert store.delete_all(user_id="u") is None

    store._enabled = True
    store._client.search.side_effect = RuntimeError("down")
    store._client.add.side_effect = RuntimeError("down")
    store._client.delete_all.side_effect = RuntimeError("down")
    assert store.search(user_id="u", query="q") == []
    assert store.add_qa(user_id="u", query="q", answer="a") is None
    assert store.delete_all(user_id="u") is None
