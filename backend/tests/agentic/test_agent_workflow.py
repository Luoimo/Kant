from __future__ import annotations

import asyncio

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk

from agents.deepread_agent import DeepReadAgent, DeepReadConfig, _build_system_msg


class SyncMemory:
    def __init__(self):
        self.setup_called = False

    def setup(self):
        self.setup_called = True


class SyncContext:
    def __init__(self, memory):
        self.memory = memory

    def __enter__(self):
        return self.memory

    def __exit__(self, exc_type, exc, tb):
        return False


class AsyncMemory:
    def __init__(self):
        self.setup_called = False

    async def setup(self):
        self.setup_called = True


class AsyncContext:
    def __init__(self, memory):
        self.memory = memory

    async def __aenter__(self):
        return self.memory

    async def __aexit__(self, exc_type, exc, tb):
        return False


class StubCheckpointStore:
    def __init__(self):
        self.sync_memory = SyncMemory()
        self.async_memory = AsyncMemory()
        self.history_calls = []
        self.clear_calls = []
        self.message_calls = []

    def create_sync_checkpointer(self):
        return SyncContext(self.sync_memory)

    def create_async_checkpointer(self):
        return AsyncContext(self.async_memory)

    def get_chat_history(self, **kwargs):
        self.history_calls.append(kwargs)
        return [{"role": "user", "content": "previous question"}]

    def clear_chat_history(self, **kwargs):
        self.clear_calls.append(kwargs)

    def add_ai_message(self, content, **kwargs):
        self.message_calls.append((content, kwargs))


class StubStore:
    collection_name = "agentic-test"


class StubReactAgent:
    def __init__(self):
        self.invoke_payload = None
        self.invoke_config = None

    def invoke(self, payload, config):
        self.invoke_payload = payload
        self.invoke_config = config
        return {"messages": [AIMessage(content="基于证据的回答")]} 

    async def astream_events(self, payload, config, version):
        self.invoke_payload = payload
        self.invoke_config = config
        assert version == "v2"
        yield {"event": "on_tool_start", "name": "search_book_content", "data": {}}
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content="第一段", tool_call_chunks=[])},
        }
        yield {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": AIMessageChunk(
                    content=[{"type": "text", "text": "第二段"}],
                    tool_call_chunks=[],
                )
            },
        }
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content="", tool_call_chunks=[{"name": "tool"}])},
        }


def _make_agent():
    checkpoint_store = StubCheckpointStore()
    agent = DeepReadAgent(
        store=StubStore(),
        checkpoint_store=checkpoint_store,
        llm=object(),
        config=DeepReadConfig(enable_graph_retrieval=False),
    )
    return agent, checkpoint_store


def _evidence_document():
    return Document(
        page_content="先验统觉把表象统一起来。",
        metadata={
            "source": "kant.epub",
            "book_title": "纯粹理性批判",
            "chapter_title": "先验分析论",
            "section_indices": "1,2",
        },
    )


def test_build_system_message_includes_runtime_context():
    message = _build_system_msg(
        book_title="纯粹理性批判",
        book_source="kant.epub",
        memory_context="用户偏好原文引用",
        selected_text="先验统觉必须能够伴随我的一切表象",
        current_chapter="先验分析论",
        locale="zh-CN",
    )

    assert "纯粹理性批判" in message
    assert "kant.epub" in message
    assert "用户偏好原文引用" in message
    assert "先验统觉必须能够伴随我的一切表象" in message
    assert "先验分析论" in message


def test_build_system_message_is_empty_without_context():
    assert _build_system_msg(None, None, "", None, None) == ""


def test_deepread_run_uses_conversation_thread_and_returns_citations(monkeypatch):
    agent, checkpoint_store = _make_agent()
    react_agent = StubReactAgent()
    docs = [_evidence_document()]

    monkeypatch.setattr(agent, "_build", lambda **kwargs: (react_agent, docs))

    result = agent.run(
        query="什么是先验统觉？",
        book_source="kant.epub",
        book_id="book-1",
        conversation_id="conversation-1",
        locale="zh-CN",
    )

    assert checkpoint_store.sync_memory.setup_called is True
    assert react_agent.invoke_payload == {"messages": [("user", "什么是先验统觉？")]}
    assert react_agent.invoke_config == {
        "configurable": {"thread_id": "conversation-1"},
        "recursion_limit": 8,
    }
    assert result.answer == "基于证据的回答"
    assert result.citations[0].source == "kant.epub"
    assert result.citations[0].chapter_title == "先验分析论"


def test_deepread_stream_translates_agent_events_and_returns_evidence(monkeypatch):
    agent, checkpoint_store = _make_agent()
    react_agent = StubReactAgent()
    docs = [_evidence_document()]
    monkeypatch.setattr(agent, "_build", lambda **kwargs: (react_agent, docs))

    async def collect_events():
        return [
            event
            async for event in agent.astream_events(
                query="解释这一段",
                book_source="kant.epub",
                book_id="book-1",
                conversation_id="conversation-2",
                locale="zh-CN",
                include_trace_details=True,
            )
        ]

    events = asyncio.run(collect_events())

    assert checkpoint_store.async_memory.setup_called is True
    assert events[0] == ("tool", "search_book_content")
    assert events[1] == (
        "tool_trace",
        {"name": "search_book_content", "input": None},
    )
    assert events[2:4] == [("token", "第一段"), ("token", "第二段")]
    assert events[-1][0] == "done"
    assert events[-1][1]["docs_count"] == 1
    assert events[-1][1]["citations"][0]["source"] == "kant.epub"
    assert events[-1][1]["retrieved_contexts"] == ["先验统觉把表象统一起来。"]
    assert react_agent.invoke_config["configurable"]["thread_id"] == "conversation-2"
    assert react_agent.invoke_config["recursion_limit"] == 8


def test_deepread_history_operations_delegate_to_checkpoint_store():
    agent, checkpoint_store = _make_agent()

    assert agent.get_chat_history(conversation_id="conversation-3") == [
        {"role": "user", "content": "previous question"}
    ]
    agent.clear_chat_history(conversation_id="conversation-3")
    agent.add_ai_message(
        "critic note",
        conversation_id="conversation-3",
        book_id="book-1",
        locale="en-US",
    )

    assert checkpoint_store.history_calls == [{"conversation_id": "conversation-3"}]
    assert checkpoint_store.clear_calls == [{"conversation_id": "conversation-3"}]
    assert checkpoint_store.message_calls[0][0] == "critic note"
    assert checkpoint_store.message_calls[0][1]["conversation_id"] == "conversation-3"
    assert callable(checkpoint_store.message_calls[0][1]["agent_builder"])
