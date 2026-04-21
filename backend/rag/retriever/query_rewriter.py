# _*_ coding:utf-8 _*_
from __future__ import annotations

import logging

from llm.openai_client import get_llm

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
你是一位专注于哲学与社会科学的信息检索专家。
用户将提供一个问题，请将其改写为更适合语义检索和关键词匹配的形式：
- 展开代词，使问题自给自足
- 补充相关哲学专业术语（中文 / 德文 / 拉丁文对应词）
- 保留原始问题的核心意图，不要过度扩展
- 只输出改写后的查询，不加任何解释或前缀
"""


class QueryRewriter:
    """使用 LLM 将用户问题改写为更适合检索的形式。"""

    def __init__(self, llm=None) -> None:
        self._llm = llm or get_llm(temperature=0.0)

    def rewrite(self, query: str) -> str:
        try:
            msg = self._llm.invoke([
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ])
            rewritten = getattr(msg, "content", str(msg)).strip()
            logger.debug("QueryRewriter: %r → %r", query, rewritten)
            return rewritten
        except Exception as exc:
            logger.warning("QueryRewriter 失败（%s），使用原始查询", exc)
            return query
