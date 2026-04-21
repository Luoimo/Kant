import logging
import json
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage, SystemMessage
from llm.openai_client import get_llm

logger = logging.getLogger(__name__)

_CRITIC_PROMPT = """你是"评论员（Critic Agent）"。
你的任务是在系统给出哲学/阅读解答后，作为独立的评估者（Evaluator）对该回答进行事实核查和客观性审查。

【你的工作原则】：
1. 幻觉检查：如果原回答宣称"书中说..."，但提供的参考文档中没有相关证据，你必须指出其可能存在幻觉（Hallucination）。
2. 公平与多元检查：如果原回答对某个哲学概念或流派的解释过于单一、带有强烈的主观偏见（Bias），你需要补充其他流派的客观视角。

【你的输出规则】：
- 如果你认为原回答非常严谨、客观且没有幻觉，你**必须**输出 "PASS"（四个大写字母），不需要任何其他字。
- 如果你发现了瑕疵或可以补充的客观视角，请写一段简短的**《审查笔记》**，使用友好的语气，例如：“💡 评论员补充：从经验主义视角来看...” 或 “⚠️ 事实核查：原文中并未明确提及这一点...”。

【注意】：
你的回复将直接追加展示给用户。如果是 PASS，前端将不显示任何审查笔记。
"""

class CriticAgent:
    """
    后置评估 Agent（Reflection / Evaluator Architecture）：
    在主回答流式输出完毕后，同步介入主流程进行最终的事实与公平性审查。
    这种非钩子（Non-hook）串联模式是 AI Agent 里的 "Supervisor/Critic" 经典设计。
    """
    def __init__(self, llm=None):
        self._llm = llm or get_llm(temperature=0.3)

    async def aevaluate(self, user_query: str, retrieved_docs_text: str, generated_answer: str) -> AsyncGenerator[str, None]:
        context = (
            f"【用户提问】：\n{user_query}\n\n"
            f"【系统检索到的证据】：\n{retrieved_docs_text}\n\n"
            f"【AI生成的回答】：\n{generated_answer}\n"
        )
        msgs = [
            SystemMessage(content=_CRITIC_PROMPT),
            HumanMessage(content=context)
        ]
        
        try:
            is_first_token = True
            is_pass = False
            
            async for chunk in self._llm.astream(msgs):
                content = chunk.content
                if not content:
                    continue
                    
                if is_first_token:
                    is_first_token = False
                    # 提前判断是否以 PASS 开头
                    if content.strip().upper().startswith("PASS"):
                        is_pass = True
                        break
                    else:
                        yield "\n\n> **🧐 审查笔记（Critic's Note）**：\n> " + content
                else:
                    if not is_pass:
                        # 对于后续 token，替换掉所有换行，确保它能乖乖待在引用框里
                        safe_content = content.replace("\n", "\n> ")
                        yield safe_content
        except Exception as e:
            logger.warning(f"CriticAgent failed: {e}")
            yield f"\n> 审查失败：{str(e)}"
