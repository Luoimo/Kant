"""Notion 后端工具：用 Notion 官方 API 替代 Obsidian CLI。

暴露三个与 obsidian_tools.py 完全同名的 @tool，保证 NoteAgent 无感切换：
- read_past_notes(book_title): 读取指定书的历史笔记
- search_vault_for_concept(query): 在整个 workspace 中搜索概念
- append_note_to_obsidian(book_title, markdown_content): 追加 Markdown 到对应书页

依赖：
    pip install notion-client

环境变量：
    NOTION_API_KEY           Notion Integration Token（必填）
    NOTION_PARENT_PAGE_ID    所有书籍页面的父页面（必填，必须把该页面共享给 Integration）

设计说明：
- 每本书对应 parent page 下的一个 child page，标题即 book_title
- 笔记以一段段 Markdown 追加进该 page 的 block children
- 为了保持兼容，append 失败时自动创建新页面
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool

from config import get_settings

logger = logging.getLogger(__name__)

try:
    from notion_client import Client as _NotionClient  # type: ignore
except Exception:  # pragma: no cover - 依赖未安装时给出友好降级
    _NotionClient = None  # type: ignore


# ---------------------------------------------------------------------------
# Client 懒加载
# ---------------------------------------------------------------------------

_client_cache: dict = {}


def _get_client():
    """惰性创建 Notion client；缺失 key 时返回 None 触发降级。"""
    if "client" in _client_cache:
        return _client_cache["client"]

    settings = get_settings()
    api_key = getattr(settings, "notion_api_key", "") or ""
    if not api_key or _NotionClient is None:
        _client_cache["client"] = None
        return None

    _client_cache["client"] = _NotionClient(auth=api_key)
    return _client_cache["client"]


def _get_parent_page_id() -> str:
    return getattr(get_settings(), "notion_parent_page_id", "") or ""


# ---------------------------------------------------------------------------
# 基础操作
# ---------------------------------------------------------------------------

def _find_book_page(title: str) -> Optional[str]:
    """在整个 workspace 中按标题搜索 page，返回 page_id 或 None。"""
    client = _get_client()
    if not client or not title:
        return None
    try:
        resp = client.search(
            query=title,
            filter={"property": "object", "value": "page"},
            page_size=10,
        )
        for item in resp.get("results", []):
            props = item.get("properties", {})
            title_prop = props.get("title") or props.get("Name") or {}
            rich = title_prop.get("title", []) if isinstance(title_prop, dict) else []
            plain = "".join(t.get("plain_text", "") for t in rich).strip()
            if plain == title:
                return item.get("id")
        # Fallback: 返回第一个（模糊匹配）
        if resp.get("results"):
            return resp["results"][0].get("id")
    except Exception as e:
        logger.warning("Notion search failed: %s", e)
    return None


def _create_book_page(title: str) -> Optional[str]:
    """在配置的 parent page 下创建子 page，返回 page_id。"""
    client = _get_client()
    parent_id = _get_parent_page_id()
    if not client or not parent_id:
        return None
    try:
        page = client.pages.create(
            parent={"page_id": parent_id},
            properties={
                "title": [{"type": "text", "text": {"content": title}}],
            },
        )
        return page.get("id")
    except Exception as e:
        logger.error("Notion create page failed: %s", e)
        return None


def _read_page_markdown(page_id: str) -> str:
    """把 page 的 block children 以简易 Markdown 形式拼回字符串。"""
    client = _get_client()
    if not client or not page_id:
        return ""
    out: list[str] = []
    cursor = None
    try:
        while True:
            kwargs = {"block_id": page_id, "page_size": 100}
            if cursor:
                kwargs["start_cursor"] = cursor
            resp = client.blocks.children.list(**kwargs)
            for blk in resp.get("results", []):
                out.append(_block_to_markdown(blk))
            if not resp.get("has_more"):
                break
            cursor = resp.get("next_cursor")
    except Exception as e:
        logger.warning("Notion read blocks failed: %s", e)
        return ""
    return "\n".join(x for x in out if x)


def _block_to_markdown(block: dict) -> str:
    btype = block.get("type", "")
    data = block.get(btype, {}) or {}
    rich = data.get("rich_text") or []
    text = "".join(r.get("plain_text", "") for r in rich)
    if btype == "heading_1":
        return f"# {text}"
    if btype == "heading_2":
        return f"## {text}"
    if btype == "heading_3":
        return f"### {text}"
    if btype == "bulleted_list_item":
        return f"- {text}"
    if btype == "numbered_list_item":
        return f"1. {text}"
    if btype == "quote":
        return f"> {text}"
    if btype == "code":
        lang = data.get("language", "")
        return f"```{lang}\n{text}\n```"
    if btype == "divider":
        return "---"
    return text


def _append_markdown(page_id: str, markdown: str) -> bool:
    """把 Markdown 文本转成简易 block 列表追加到页面末尾。"""
    client = _get_client()
    if not client or not page_id:
        return False
    blocks = _markdown_to_blocks(markdown)
    if not blocks:
        return False
    try:
        # Notion 一次最多 100 个 block
        for i in range(0, len(blocks), 90):
            client.blocks.children.append(
                block_id=page_id,
                children=blocks[i : i + 90],
            )
        return True
    except Exception as e:
        logger.error("Notion append blocks failed: %s", e)
        return False


def _markdown_to_blocks(md: str) -> list[dict]:
    """把 Markdown 字符串按行映射成 Notion block。

    只做轻量转换以保证写入不报错；复杂场景可以后续升级成完整 parser。
    """
    blocks: list[dict] = []
    lines = md.splitlines()
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            blocks.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": []},
                }
            )
            continue
        if line.startswith("### "):
            blocks.append(_heading_block(line[4:], level=3))
        elif line.startswith("## "):
            blocks.append(_heading_block(line[3:], level=2))
        elif line.startswith("# "):
            blocks.append(_heading_block(line[2:], level=1))
        elif line.startswith("- ") or line.startswith("* "):
            blocks.append(_list_block(line[2:], numbered=False))
        elif line.lstrip().startswith(("1. ", "2. ", "3. ")):
            blocks.append(_list_block(line.split(". ", 1)[1], numbered=True))
        elif line.startswith("> "):
            blocks.append(
                {
                    "object": "block",
                    "type": "quote",
                    "quote": {"rich_text": _rich(line[2:])},
                }
            )
        else:
            blocks.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": _rich(line)},
                }
            )
    return blocks


def _heading_block(text: str, level: int) -> dict:
    key = f"heading_{level}"
    return {
        "object": "block",
        "type": key,
        key: {"rich_text": _rich(text)},
    }


def _list_block(text: str, *, numbered: bool) -> dict:
    key = "numbered_list_item" if numbered else "bulleted_list_item"
    return {
        "object": "block",
        "type": key,
        key: {"rich_text": _rich(text)},
    }


def _rich(text: str) -> list[dict]:
    if not text:
        return []
    # Notion 单段 rich_text 限制 2000 字符
    chunks = [text[i : i + 1900] for i in range(0, len(text), 1900)]
    return [{"type": "text", "text": {"content": c}} for c in chunks]


# ---------------------------------------------------------------------------
# LangChain Tools（保持与 obsidian_tools.py 同名签名）
# ---------------------------------------------------------------------------


@tool
def read_past_notes(book_title: str) -> str:
    """读取当前书籍在 Notion 知识库中的历史笔记全文。"""
    if not _get_client():
        return "Error: Notion 未配置 (缺少 NOTION_API_KEY)"
    page_id = _find_book_page(book_title)
    if not page_id:
        return f"暂无《{book_title}》的历史笔记"
    content = _read_page_markdown(page_id)
    return content or f"《{book_title}》页面为空"


@tool
def search_vault_for_concept(query: str) -> str:
    """在整个 Notion workspace 中搜索某个概念或关键词，返回最多 5 条结果摘要。"""
    client = _get_client()
    if not client:
        return "Error: Notion 未配置 (缺少 NOTION_API_KEY)"
    try:
        resp = client.search(
            query=query,
            filter={"property": "object", "value": "page"},
            page_size=5,
        )
    except Exception as e:
        logger.warning("Notion search failed: %s", e)
        return f"Error: {e}"

    lines: list[str] = []
    for item in resp.get("results", []):
        props = item.get("properties", {})
        title_prop = props.get("title") or props.get("Name") or {}
        rich = title_prop.get("title", []) if isinstance(title_prop, dict) else []
        title = "".join(t.get("plain_text", "") for t in rich).strip() or "(无标题)"
        url = item.get("url", "")
        lines.append(f"- {title} :: {url}")
    return "\n".join(lines) if lines else f"未在 Notion 中搜索到与 '{query}' 相关的页面"


@tool
def append_note_to_obsidian(book_title: str, markdown_content: str) -> str:
    """将整理好的 Markdown 笔记追加写入对应书籍的 Notion 页面；不存在则新建。

    （保持与 Obsidian 版同名，避免 NoteAgent prompt/调用逻辑改动。）
    """
    if not _get_client():
        return "Error: Notion 未配置 (缺少 NOTION_API_KEY)"

    content = markdown_content.strip()
    if not content:
        return "Error: 内容为空"

    page_id = _find_book_page(book_title)
    if not page_id:
        page_id = _create_book_page(book_title)
        if not page_id:
            return "Error: 创建 Notion 页面失败 (检查 NOTION_PARENT_PAGE_ID 是否已共享给 Integration)"

    ok = _append_markdown(page_id, content)
    if not ok:
        return "Error: 追加 Notion 内容失败"
    return f"已写入 Notion 页面《{book_title}》"


TOOLS = [read_past_notes, search_vault_for_concept, append_note_to_obsidian]
