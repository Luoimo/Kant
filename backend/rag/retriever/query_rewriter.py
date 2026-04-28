# _*_ coding:utf-8 _*_
from __future__ import annotations

import logging

from llm.openai_client import get_llm
from prompts import get_prompts

logger = logging.getLogger(__name__)


class QueryRewriter:
    """使用 LLM 将用户问题改写为更适合检索的形式。"""

    def __init__(self, llm=None, *, locale: str | None = None) -> None:
        self._llm = llm or get_llm(temperature=0.0)
        self._locale = locale

    def rewrite(self, query: str, *, locale: str | None = None) -> str:
        eff_locale = locale or self._locale
        system_prompt = get_prompts(eff_locale).retriever.query_rewriter_system
        try:
            msg = self._llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ])
            rewritten = getattr(msg, "content", str(msg)).strip()
            logger.debug("QueryRewriter: %r → %r", query, rewritten)
            return rewritten
        except Exception as exc:
            logger.warning("QueryRewriter failed (%s), using original query", exc)
            return query
