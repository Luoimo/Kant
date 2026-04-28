import json
import logging
import re
from langchain_core.messages import HumanMessage, SystemMessage
from llm.openai_client import get_llm
from prompts import get_prompts

logger = logging.getLogger(__name__)


class FollowupAgent:
    """
    Hook agent: suggests follow-up questions after each Q&A round.
    """
    def __init__(self, llm=None):
        self._llm = llm or get_llm(temperature=0.4)

    def generate(self, question: str, answer: str, *, locale: str | None = None) -> list[str]:
        p = get_prompts(locale).followup
        msgs = [
            SystemMessage(content=p.system),
            HumanMessage(content=p.input_template.format(question=question, answer=answer)),
        ]
        try:
            resp = self._llm.invoke(msgs)
            return self._parse_response(resp.content)
        except Exception as e:
            logger.warning(f"Followup generation failed: {e}")
            return []

    async def agenerate(self, question: str, answer: str, *, locale: str | None = None) -> list[str]:
        p = get_prompts(locale).followup
        msgs = [
            SystemMessage(content=p.system),
            HumanMessage(content=p.input_template.format(question=question, answer=answer)),
        ]
        try:
            resp = await self._llm.ainvoke(msgs)
            return self._parse_response(resp.content)
        except Exception as e:
            logger.warning(f"Async Followup generation failed: {e}")
            return []

    def _parse_response(self, content: str) -> list[str]:
        text = content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(q)[:100] for q in data][:3]
        except json.JSONDecodeError:
            pass
        return []
