from __future__ import annotations

import logging
import subprocess
from typing import List

from langchain_core.tools import tool

from config import get_settings

logger = logging.getLogger(__name__)

def _run_obsidian(args: List[str]) -> str:
    """Helper to run obsidian cli with the configured vault."""
    settings = get_settings()
    cmd = ["obsidian"] + args
    if hasattr(settings, "obsidian_vault") and settings.obsidian_vault:
        cmd.append(f"vault={settings.obsidian_vault}")
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        err = e.stderr.strip() or e.stdout.strip()
        logger.warning(f"Obsidian CLI command failed: {' '.join(cmd)} | Error: {err}")
        return f"Error: {err}"
    except Exception as e:
        logger.error(f"Failed to execute Obsidian CLI: {e}")
        return f"Error: {e}"

@tool
def read_past_notes(book_title: str) -> str:
    """读取当前书籍在 Obsidian 知识库中的历史笔记全文。
    这能帮助你了解之前记过什么，从而避免重复，或接续之前的思路。
    """
    # Using file=book_title (without .md since obsidian cli handles resolution)
    return _run_obsidian(["read", f"file={book_title}"])

@tool
def search_vault_for_concept(query: str) -> str:
    """在整个 Obsidian 知识库中搜索某个概念或关键词。
    如果新笔记提到了某个重要的哲学概念，你可以用这个工具看看库里的其他书籍是否也提到过，
    从而在整理笔记时使用双向链接将它们串联起来。
    """
    return _run_obsidian(["search", f"query={query}", "limit=5"])

@tool
def append_note_to_obsidian(book_title: str, markdown_content: str) -> str:
    """将你精心整理并带有双向链接的 Markdown 笔记追加写入到对应书籍的 Obsidian 文件中。
    注意：在完成所有思考和内容编排后，必须调用此工具将笔记持久化。
    """
    # 先尝试追加
    res = _run_obsidian(["append", f"file={book_title}", f"content={markdown_content}", "inline"])
    
    # 如果文件不存在，则创建新文件
    if "not found" in res.lower() or "Error" in res:
        res = _run_obsidian(["create", f"name={book_title}", f"content={markdown_content}"])
        
    return res

# 暴露给外部使用
TOOLS = [read_past_notes, search_vault_for_concept, append_note_to_obsidian]
