import json
import logging
import re
from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from llm.openai_client import get_llm
from prompts import get_prompts

logger = logging.getLogger(__name__)


class RouterAgent:
    """
    Front-end routing agent (Router Architecture):
    classifies the user query before the main answering pipeline starts.
    """
    def __init__(self, llm=None):
        self._llm = llm or get_llm(temperature=0.1)

    async def aroute(self, user_query: str, *, locale: str | None = None) -> Dict[str, Any]:
        p = get_prompts(locale).router
        # Build a small language-aware user wrapper.
        user_wrapper = (
            f"用户提问：{user_query}" if p.intent_status == "正在识别意图…"
            else f"User question: {user_query}"
        )
        msgs = [
            SystemMessage(content=p.system),
            HumanMessage(content=user_wrapper),
        ]
        try:
            resp = await self._llm.ainvoke(msgs)
            text = resp.content.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            data = json.loads(text)

            return {
                "intent": data.get("intent", "book_qa"),
                "optimized_query": user_query,
            }
        except Exception as e:
            logger.warning(f"RouterAgent parse failed: {e}")
            return {"intent": "book_qa", "optimized_query": user_query}
