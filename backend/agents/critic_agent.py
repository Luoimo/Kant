import logging
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage, SystemMessage
from llm.openai_client import get_llm
from prompts import get_prompts

logger = logging.getLogger(__name__)


class CriticAgent:
    """
    Post-hoc evaluator (Reflection / Evaluator architecture):
    After the main answer is streamed, synchronously inspect it for hallucination
    and fairness.
    """
    def __init__(self, llm=None):
        self._llm = llm or get_llm(temperature=0.3)

    async def aevaluate(
        self,
        user_query: str,
        retrieved_docs_text: str,
        generated_answer: str,
        *,
        locale: str | None = None,
    ) -> AsyncGenerator[str, None]:
        p = get_prompts(locale).critic
        context = (
            f"{p.input_header_query}:\n{user_query}\n\n"
            f"{p.input_header_evidence}:\n{retrieved_docs_text}\n\n"
            f"{p.input_header_answer}:\n{generated_answer}\n"
        )
        msgs = [
            SystemMessage(content=p.system),
            HumanMessage(content=context),
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
                    # Early detection of "PASS" prefix (short-circuit)
                    if content.strip().upper().startswith("PASS"):
                        is_pass = True
                        break
                    else:
                        yield p.critic_note_title + content
                else:
                    if not is_pass:
                        # Keep later tokens inside the blockquote.
                        safe_content = content.replace("\n", "\n> ")
                        yield safe_content
        except Exception as e:
            logger.warning(f"CriticAgent failed: {e}")
            yield p.critic_failed.format(err=str(e))
