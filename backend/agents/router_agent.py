import json
import logging
import re
from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from llm.openai_client import get_llm

logger = logging.getLogger(__name__)

_ROUTER_PROMPT = """你是"路由主管（Router Agent）"。
你的任务是在主流程开始前，对用户的提问进行意图分类。

请将用户的输入分为以下两类之一：
1. "book_qa" : 询问书本知识、哲学概念、阅读建议等需要专业深读的问题。
2. "casual"  : 闲聊、打招呼等不需要查询书库的问题。

必须输出如下格式的纯 JSON（不要带代码块或多余文字）：
{
  "intent": "book_qa" 或 "casual"
}
"""

class RouterAgent:
    """
    前端路由 Agent（Router Architecture）：
    负责在进入主回答逻辑前，对 Query 进行意图识别。
    """
    def __init__(self, llm=None):
        self._llm = llm or get_llm(temperature=0.1)

    async def aroute(self, user_query: str) -> Dict[str, Any]:
        msgs = [
            SystemMessage(content=_ROUTER_PROMPT),
            HumanMessage(content=f"用户提问：{user_query}")
        ]
        try:
            resp = await self._llm.ainvoke(msgs)
            text = resp.content.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            data = json.loads(text)
            
            return {
                "intent": data.get("intent", "book_qa"),
                "optimized_query": user_query # 保持原样，不再重写
            }
        except Exception as e:
            logger.warning(f"RouterAgent 解析失败: {e}")
            return {"intent": "book_qa", "optimized_query": user_query}
