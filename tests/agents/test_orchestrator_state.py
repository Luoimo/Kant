"""Tests for the new Agent Team orchestration layer.

Replaces tests that covered orchestrator_agent.py (now deleted).
Tests cover: MessageEnvelope, WorkerPool, Dispatcher payload building,
and _extract_book_from_answer utility.
"""
from __future__ import annotations

import queue
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest

from backend.team.message import MessageEnvelope, VALID_MSG_TYPES
from backend.team.worker import AgentWorker, WorkerPool
from backend.team.dispatcher import (
    DispatchResult,
    IntentGraph,
    _build_payload,
    _collect,
    _extract_book_from_answer,
)


# ---------------------------------------------------------------------------
# MessageEnvelope
# ---------------------------------------------------------------------------

def test_message_envelope_valid_types():
    for msg_type in VALID_MSG_TYPES:
        env = MessageEnvelope(
            msg_type=msg_type,
            request_id=uuid.uuid4().hex,
            sender="test",
            payload={},
        )
        assert env.msg_type == msg_type


def test_message_envelope_rejects_invalid_type():
    with pytest.raises(ValueError, match="Invalid msg_type"):
        MessageEnvelope(
            msg_type="unknown_type",
            request_id="abc",
            sender="test",
            payload={},
        )


def test_message_envelope_reply_to_defaults_none():
    env = MessageEnvelope(msg_type="task_request", request_id="x", sender="s", payload={})
    assert env.reply_to is None


# ---------------------------------------------------------------------------
# AgentWorker — task_request and shutdown FSM
# ---------------------------------------------------------------------------

class EchoWorker(AgentWorker):
    """Test worker that echoes its payload back as the result."""
    def __init__(self):
        super().__init__(name="echo", role="test")

    def _execute(self, payload):
        return {"echo": payload.get("value", ""), "intent": "echo"}


def test_worker_processes_task_request():
    worker = EchoWorker()
    worker.start()
    reply_q: queue.Queue[MessageEnvelope] = queue.Queue()
    req_id = uuid.uuid4().hex

    worker.inbox.put(MessageEnvelope(
        msg_type="task_request",
        request_id=req_id,
        sender="test",
        payload={"value": "hello"},
        reply_to=reply_q,
    ))

    resp = reply_q.get(timeout=5.0)
    assert resp.msg_type == "task_response"
    assert resp.request_id == req_id
    assert resp.payload["echo"] == "hello"

    # Cleanup
    worker.inbox.put(MessageEnvelope(
        msg_type="shutdown_request", request_id="x", sender="test",
        payload={}, reply_to=queue.Queue(),
    ))
    worker.join(timeout=3.0)


def test_worker_returns_task_error_on_exception():
    class FailWorker(AgentWorker):
        def __init__(self):
            super().__init__(name="fail", role="test")
        def _execute(self, payload):
            raise ValueError("intentional failure")

    worker = FailWorker()
    worker.start()
    reply_q: queue.Queue[MessageEnvelope] = queue.Queue()
    req_id = uuid.uuid4().hex

    worker.inbox.put(MessageEnvelope(
        msg_type="task_request", request_id=req_id, sender="test",
        payload={}, reply_to=reply_q,
    ))

    resp = reply_q.get(timeout=5.0)
    assert resp.msg_type == "task_error"
    assert "intentional failure" in resp.payload["error"]

    worker.inbox.put(MessageEnvelope(
        msg_type="shutdown_request", request_id="x", sender="test",
        payload={}, reply_to=queue.Queue(),
    ))
    worker.join(timeout=3.0)


def test_worker_shutdown_fsm():
    worker = EchoWorker()
    worker.start()
    reply_q: queue.Queue[MessageEnvelope] = queue.Queue()
    req_id = uuid.uuid4().hex

    worker.inbox.put(MessageEnvelope(
        msg_type="shutdown_request", request_id=req_id, sender="test",
        payload={}, reply_to=reply_q,
    ))

    resp = reply_q.get(timeout=5.0)
    assert resp.msg_type == "shutdown_response"
    assert resp.payload["approve"] is True
    worker.join(timeout=3.0)
    assert not worker.is_alive()


# ---------------------------------------------------------------------------
# WorkerPool
# ---------------------------------------------------------------------------

def test_worker_pool_creates_correct_number_of_workers():
    pool = WorkerPool("test", lambda: EchoWorker(), size=3)
    assert len(pool.all_workers()) == 3


def test_worker_pool_least_busy_returns_idle_first():
    pool = WorkerPool("test", lambda: EchoWorker(), size=2)
    pool.workers[0].status = "working"
    pool.workers[1].status = "idle"
    assert pool.least_busy() is pool.workers[1]


def test_worker_pool_least_busy_falls_back_to_qsize():
    pool = WorkerPool("test", lambda: EchoWorker(), size=2)
    pool.workers[0].status = "working"
    pool.workers[1].status = "working"
    # Both working; pool falls back to qsize (both 0, so picks first)
    result = pool.least_busy()
    assert result in pool.workers


# ---------------------------------------------------------------------------
# _extract_book_from_answer
# ---------------------------------------------------------------------------

def test_extract_book_finds_first_guillemet():
    answer = "推荐阅读《纯粹理性批判》，也可参考《实践理性批判》。"
    assert _extract_book_from_answer(answer) == "纯粹理性批判"


def test_extract_book_returns_empty_when_no_guillemet():
    assert _extract_book_from_answer("没有书名") == ""


def test_extract_book_handles_empty_string():
    assert _extract_book_from_answer("") == ""


# ---------------------------------------------------------------------------
# _build_payload
# ---------------------------------------------------------------------------

def _book_meta(title="测试书", source="data/books/test.epub", book_id="abc123"):
    return {"title": title, "source": source, "book_id": book_id}


def test_build_payload_deepread_includes_book_source():
    payload = _build_payload(
        "deepread", "什么是先验感性论", _book_meta(), "mem", "t1", None, None, {}
    )
    assert payload["book_source"] == "data/books/test.epub"
    assert payload["book_title"] == "测试书"
    assert payload["memory_context"] == "mem"


def test_build_payload_deepread_injects_selected_text():
    payload = _build_payload(
        "deepread", "解释一下这段话", _book_meta(), "", "t1",
        "原文片段内容", None, {}
    )
    assert "原文片段内容" in payload["query"]


def test_build_payload_deepread_injects_chapter():
    payload = _build_payload(
        "deepread", "这章讲什么", _book_meta(), "", "t1",
        None, "第三章：先验分析论", {}
    )
    assert "第三章" in payload["query"]


def test_build_payload_plan_extracts_upstream_recommend():
    upstream = {"recommend": {"answer": "推荐阅读《批判的哲学》，此书…"}}
    payload = _build_payload(
        "plan", "制定计划", None, "", "t1", None, None, upstream
    )
    assert payload["book_title"] == "批判的哲学"


def test_build_payload_plan_uses_book_meta_when_available():
    payload = _build_payload(
        "plan", "修改计划", _book_meta(), "", "t1", None, None, {}
    )
    assert payload["book_title"] == "测试书"
    assert payload["book_id"] == "abc123"


def test_build_payload_recommend_includes_current_book():
    payload = _build_payload(
        "recommend", "推荐类似书", _book_meta(), "", "t1", None, None, {}
    )
    assert payload["current_book"] == "测试书"


# ---------------------------------------------------------------------------
# _collect
# ---------------------------------------------------------------------------

def test_collect_returns_all_responses_within_timeout():
    q: queue.Queue[MessageEnvelope] = queue.Queue()
    expected = {"req1": "deepread", "req2": "recommend"}

    q.put(MessageEnvelope(msg_type="task_response", request_id="req1", sender="d", payload={}))
    q.put(MessageEnvelope(msg_type="task_response", request_id="req2", sender="r", payload={}))

    result = _collect(q, expected, timeout=2.0)
    assert set(result.keys()) == {"req1", "req2"}


def test_collect_returns_partial_on_timeout():
    q: queue.Queue[MessageEnvelope] = queue.Queue()
    expected = {"req1": "deepread", "req2": "recommend"}

    q.put(MessageEnvelope(msg_type="task_response", request_id="req1", sender="d", payload={}))
    # req2 never arrives

    result = _collect(q, expected, timeout=0.1)
    assert "req1" in result
    assert "req2" not in result


def test_collect_empty_on_full_timeout():
    q: queue.Queue[MessageEnvelope] = queue.Queue()
    result = _collect(q, {"req1": "deepread"}, timeout=0.05)
    assert result == {}


# ---------------------------------------------------------------------------
# DispatchResult
# ---------------------------------------------------------------------------

def test_dispatch_result_to_dict():
    r = DispatchResult(answer="test", citations=[{"a": 1}], retrieved_docs_count=3, intent="deepread")
    d = r.to_dict()
    assert d["answer"] == "test"
    assert d["citations"] == [{"a": 1}]
    assert d["retrieved_docs_count"] == 3
    assert d["intent"] == "deepread"
    assert "blocked" not in d
