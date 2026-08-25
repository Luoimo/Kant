from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

from langchain_core.messages import AIMessage, AIMessageChunk

from agents.critic_agent import CriticAgent
from agents.followup_agent import FollowupAgent
from agents.router_agent import RouterAgent


CASES_PATH = Path(__file__).with_name("cases.json")


class StubLLM:
    def __init__(self, *, content: str = "", chunks: list[str] | None = None, error: Exception | None = None):
        self.content = content
        self.chunks = chunks or []
        self.error = error
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        return AIMessage(content=self.content)

    async def ainvoke(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        return AIMessage(content=self.content)

    async def astream(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        for content in self.chunks:
            yield AIMessageChunk(content=content)


def _run(coro):
    return asyncio.run(coro)


def _collect(async_iterable):
    async def collect():
        return [item async for item in async_iterable]

    return _run(collect())


def test_golden_dataset_has_expected_size_and_categories():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    assert len(cases) == 30
    assert len({case["id"] for case in cases}) == 30
    assert Counter(case["category"] for case in cases) == {
        "routing": 5,
        "trajectory": 5,
        "grounding": 8,
        "security": 6,
        "memory": 4,
        "recovery": 2,
    }
    assert all(case["input"].strip() for case in cases)
    assert all(case["expected"] for case in cases)


def test_router_parses_fenced_book_qa_response():
    llm = StubLLM(content='```json\n{"intent": "book_qa"}\n```')

    result = _run(RouterAgent(llm=llm).aroute("解释先验统觉", locale="zh-CN"))

    assert result == {"intent": "book_qa", "optimized_query": "解释先验统觉"}
    assert llm.messages[-1].content == "用户提问：解释先验统觉"


def test_router_parses_english_casual_response():
    llm = StubLLM(content='{"intent": "casual"}')

    result = _run(RouterAgent(llm=llm).aroute("Hello", locale="en-US"))

    assert result == {"intent": "casual", "optimized_query": "Hello"}
    assert llm.messages[-1].content == "User question: Hello"


def test_router_defaults_missing_intent_to_book_qa():
    result = _run(RouterAgent(llm=StubLLM(content="{}")).aroute("解释这一章"))

    assert result["intent"] == "book_qa"


def test_router_rejects_unknown_intent():
    result = _run(RouterAgent(llm=StubLLM(content='{"intent": "delete_all"}')).aroute("删除所有内容"))

    assert result["intent"] == "book_qa"


def test_router_falls_back_on_invalid_json():
    result = _run(RouterAgent(llm=StubLLM(content="not-json")).aroute("解释这一章"))

    assert result == {"intent": "book_qa", "optimized_query": "解释这一章"}


def test_router_falls_back_on_llm_error():
    llm = StubLLM(error=RuntimeError("router unavailable"))

    result = _run(RouterAgent(llm=llm).aroute("解释这一章"))

    assert result == {"intent": "book_qa", "optimized_query": "解释这一章"}


def test_followup_parses_fenced_json_and_limits_output():
    long_question = "问题" * 60
    agent = FollowupAgent(llm=StubLLM())

    result = agent._parse_response(
        f'```json\n["问题一", "问题二", "{long_question}", "不会返回的问题"]\n```'
    )

    assert result[:2] == ["问题一", "问题二"]
    assert len(result) == 3
    assert len(result[2]) == 100


def test_followup_rejects_non_list_json():
    assert FollowupAgent(llm=StubLLM())._parse_response('{"question": "下一步？"}') == []


def test_followup_generate_uses_localized_prompt():
    llm = StubLLM(content='["下一步研究什么？"]')

    result = FollowupAgent(llm=llm).generate("问题", "回答", locale="zh-CN")

    assert result == ["下一步研究什么？"]
    assert "用户问题：\n问题" in llm.messages[-1].content
    assert "AI回答：\n回答" in llm.messages[-1].content


def test_followup_generate_returns_empty_list_on_error():
    agent = FollowupAgent(llm=StubLLM(error=RuntimeError("follow-up unavailable")))

    assert agent.generate("question", "answer") == []


def test_followup_agenerate_returns_questions():
    agent = FollowupAgent(llm=StubLLM(content='["Why?", "What follows?"]'))

    result = _run(agent.agenerate("Question", "Answer", locale="en-US"))

    assert result == ["Why?", "What follows?"]


def test_followup_agenerate_returns_empty_list_on_error():
    agent = FollowupAgent(llm=StubLLM(error=RuntimeError("follow-up unavailable")))

    assert _run(agent.agenerate("Question", "Answer")) == []


def test_critic_hides_pass_result():
    agent = CriticAgent(llm=StubLLM(chunks=["PASS"]))

    assert _collect(agent.aevaluate("问题", "证据", "回答")) == []


def test_critic_streams_review_note_and_formats_newlines():
    llm = StubLLM(chunks=["原回答缺少证据。", "\n请补充引用。"])

    result = _collect(CriticAgent(llm=llm).aevaluate("问题", "证据", "回答", locale="zh-CN"))

    assert result[0].startswith("\n\n> **🧐 审查笔记")
    assert result[0].endswith("原回答缺少证据。")
    assert result[1] == "\n> 请补充引用。"
    assert "【系统检索到的证据】:\n证据" in llm.messages[-1].content


def test_critic_reports_failure_without_raising():
    agent = CriticAgent(llm=StubLLM(error=RuntimeError("critic unavailable")))

    result = _collect(agent.aevaluate("Question", "Evidence", "Answer", locale="en-US"))

    assert result == ["\n> Review failed: critic unavailable"]
