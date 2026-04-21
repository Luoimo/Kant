import json
import logging
import re
from langchain_core.messages import HumanMessage, SystemMessage
from llm.openai_client import get_llm

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """你是读书引导助手。
每次用户与AI进行了一轮关于书籍内容的问答后，你需要根据这轮问答的内容，提出3个相关的延伸追问问题。
这些问题需要：
1. 能够引导用户更深入地思考。
2. 简短、直接、具有启发性。
3. 必须输出合法的JSON数组，如 ["问题1", "问题2", "问题3"]，不要包含任何其他说明文字或代码块标记。"""

class FollowupAgent:
    """
    Hook agent: 追问 Agent。
    在一轮问答结束后，根据对话内容生成3个相关追问问题，返回给前端供用户点击。
    """
    def __init__(self, llm=None):
        self._llm = llm or get_llm(temperature=0.4)

    def generate(self, question: str, answer: str) -> list[str]:
        msgs = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"用户问题：\n{question}\n\nAI回答：\n{answer}")
        ]
        try:
            resp = self._llm.invoke(msgs)
            return self._parse_response(resp.content)
        except Exception as e:
            logger.warning(f"Followup generation failed: {e}")
            return []

    async def agenerate(self, question: str, answer: str) -> list[str]:
        msgs = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"用户问题：\n{question}\n\nAI回答：\n{answer}")
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
