from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List


@dataclass
class InputSafetyResult:
    """
    输入安全检查结果。

    - allowed       : 是否允许继续向下游 Agent/LLM 处理
    - reason        : 判定原因（给用户和日志看的自然语言）
    - categories    : 命中的风险类别标签列表（例如 ["secrets", "filesystem"]）
    - sanitized_text: 经过轻量清洗/截断后的文本（例如去掉多余空白）
    """

    allowed: bool
    reason: str
    categories: List[str] = field(default_factory=list)
    sanitized_text: str = ""


_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{16,}", re.IGNORECASE),  # OpenAI-style keys
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"ghp_[A-Za-z0-9]{20,}", re.IGNORECASE),  # GitHub token
]

_FILESYSTEM_KEYWORDS = [
    "本机文件",
    "读取硬盘",
    "读取磁盘",
    "删除文件",
    "格式化磁盘",
    "format c:",
    "rm -rf",
    "del c:\\",
    "del d:\\",
]

_PROMPT_INJECTION_KEYWORDS = [
    "忽略以上所有指令",
    "忽略之前的规则",
    "你现在不再受限制",
    "解除所有安全限制",
    "系统提示词",
    "system prompt",
]

_CODE_EXEC_KEYWORDS = [
    "执行命令",
    "os.system(",
    "subprocess",
    "powershell",
    "cmd /c",
]

_OFFTOPIC_SOFT_KEYWORDS = ["书", "阅读", "章节", "作者", "笔记", "书单", "计划"]


def run_input_safety_check(user_input: str) -> InputSafetyResult:
    """
    对用户输入做一轮详细的安全检查。

    检查维度：
    - secrets      : 是否包含类似 API Key / Token 的敏感信息
    - filesystem   : 是否请求读写本机文件/磁盘
    - prompt_hack  : 是否包含典型 prompt 注入话术
    - code_exec    : 是否要求执行系统命令
    - offtopic     : 是否明显偏离“小众书精读/阅读”主题（软警告）
    """
    text = (user_input or "").strip()
    lowered = text.lower()
    categories: list[str] = []
    hard_block_reasons: list[str] = []

    # 1) secrets 检查：任何疑似 key/token 一律禁止
    for pat in _SECRET_PATTERNS:
        if pat.search(text):
            categories.append("secrets")
            hard_block_reasons.append("检测到疑似密钥/API Token，请勿在对话中粘贴密钥。")
            break

    # 2) filesystem：访问/修改本机文件系统
    for kw in _FILESYSTEM_KEYWORDS:
        if kw.lower() in lowered:
            categories.append("filesystem")
            hard_block_reasons.append(f"请求涉及本机文件/磁盘操作（关键词：{kw}），当前助手不支持。")
            break

    # 3) prompt 注入话术
    for kw in _PROMPT_INJECTION_KEYWORDS:
        if kw.lower() in lowered:
            categories.append("prompt_hack")
            hard_block_reasons.append("检测到试图覆盖/绕过系统规则的指令，已拒绝执行。")
            break

    # 4) 代码/命令执行
    for kw in _CODE_EXEC_KEYWORDS:
        if kw.lower() in lowered:
            categories.append("code_exec")
            hard_block_reasons.append("当前助手仅用于读书精读，不支持执行系统命令/代码。")
            break

    # 如果命中任意一条硬阻断规则，则直接拒绝
    if hard_block_reasons:
        reason = "；".join(hard_block_reasons)
        return InputSafetyResult(
            allowed=False,
            reason=reason,
            categories=categories,
            sanitized_text=text,
        )

    # 5) 主题相关性（软警告）：不强制拦截，但提醒用户
    if not any(k in text for k in _OFFTOPIC_SOFT_KEYWORDS):
        categories.append("offtopic")
        reason = "安全检查通过，但当前请求与“小众书精读/阅读”主题关联度较弱，请确认是否是你的本意。"
    else:
        reason = "安全检查通过。"

    return InputSafetyResult(
        allowed=True,
        reason=reason,
        categories=categories,
        sanitized_text=text,
    )

