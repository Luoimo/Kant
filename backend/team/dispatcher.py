"""Dispatcher — intent classification, staged parallel execution, result aggregation.

Replaces run_minimal_graph() from orchestrator_agent.py.
Implements the IntentGraph staged execution pattern from the spec,
inspired by s07 task dependency graph and s10 request_id correlation.
"""
from __future__ import annotations

import json
import queue
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from backend.llm.openai_client import get_llm
from backend.memory.mem0_store import Mem0Store
from backend.security.input_filter import run_input_safety_check
from backend.storage.book_catalog import get_book_catalog
from backend.team.message import MessageEnvelope
from backend.team.team import AgentTeam


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieved_docs_count: int = 0
    intent: str = ""
    blocked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "retrieved_docs_count": self.retrieved_docs_count,
            "intent": self.intent,
        }


@dataclass
class IntentGraph:
    stages: list[list[str]]


# ---------------------------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = """你是意图分类器。分析用户输入，返回JSON格式的意图图。

可用意图：
- deepread: 深度解读书籍内容、提问书中知识、理解书中概念
- plan: 修改、更新或扩展阅读计划（需要书籍已打开）
- recommend: 推荐书籍

规则：
1. 没有依赖关系的意图放在同一阶段（并行执行）
2. plan 或 notes 依赖 recommend 输出时（如"先推荐再制定计划"），分为两个阶段
3. 只输出JSON，不要有其他内容

示例：
用户问书中知识 → {"stages": [["deepread"]]}
推荐书籍 → {"stages": [["recommend"]]}
同时解读+推荐 → {"stages": [["deepread", "recommend"]]}
先推荐再制定计划 → {"stages": [["recommend"], ["plan"]]}
修改计划 → {"stages": [["plan"]]}"""


def _classify_intent(query: str, book_title: str, active_tab: str | None) -> IntentGraph:
    """Return an IntentGraph. Falls back to [["deepread"]] on any failure."""
    if active_tab and active_tab in {"deepread", "plan", "recommend", "notes"}:
        return IntentGraph(stages=[[active_tab]])

    llm = get_llm(temperature=0.0)
    context = f"当前阅读书籍：《{book_title}》\n\n" if book_title else ""
    user_msg = f"{context}用户输入：{query}"

    try:
        msg = llm.invoke([
            {"role": "system", "content": _CLASSIFIER_SYSTEM},
            {"role": "user", "content": user_msg},
        ])
        raw = getattr(msg, "content", str(msg)).strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        data = json.loads(raw)
        stages = data.get("stages", [])
        # Validate: each stage is a non-empty list of known intents
        valid = {"deepread", "plan", "recommend", "notes"}
        cleaned: list[list[str]] = []
        for stage in stages:
            s = [i for i in stage if i in valid]
            if s:
                cleaned.append(s)
        if cleaned:
            return IntentGraph(stages=cleaned)
    except Exception as exc:
        print(f"[Dispatcher] intent classification failed: {exc}", file=sys.stderr)

    return IntentGraph(stages=[["deepread"]])


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def _extract_book_from_answer(answer: str) -> str:
    """Extract first 《书名》 from recommendation text."""
    m = re.search(r"《([^》]+)》", answer)
    return m.group(1) if m else ""


def _build_payload(
    intent: str,
    query: str,
    book_meta: dict | None,
    memory_ctx: str,
    thread_id: str,
    selected_text: str | None,
    current_chapter: str | None,
    upstream: dict[str, dict],
) -> dict[str, Any]:
    book_source = book_meta["source"] if book_meta else None
    book_title = book_meta.get("title", "") if book_meta else ""
    book_id = book_meta.get("book_id", "") if book_meta else ""

    # Inject reader context into the query for deepread
    task_query = query
    if intent == "deepread":
        if book_title:
            task_query = f"【当前阅读书籍】：《{book_title}》\n\n{task_query}"
        if selected_text:
            task_query = f"【用户划选的原文片段】：\n{selected_text}\n\n【用户问题】：\n{task_query}"
        if current_chapter:
            task_query += f"\n\n【当前阅读章节】：{current_chapter}"

    if intent == "deepread":
        return {
            "query": task_query,
            "book_source": book_source,
            "book_title": book_title,
            "book_id": book_id,
            "memory_context": memory_ctx,
            "thread_id": thread_id,
        }

    if intent in ("plan",):
        # If upstream has recommend output, extract the recommended book
        if "recommend" in upstream and not book_title:
            rec_answer = upstream["recommend"].get("answer", "")
            book_title = _extract_book_from_answer(rec_answer)
        return {
            "query": query,
            "book_title": book_title,
            "book_id": book_id,
            "action": "edit",
            "memory_context": memory_ctx,
            "thread_id": thread_id,
        }

    if intent == "recommend":
        return {
            "query": query,
            "current_book": book_title,
            "memory_context": memory_ctx,
            "recommend_type": "discover",
            "thread_id": thread_id,
        }

    if intent == "notes":
        return {"query": query, "thread_id": thread_id}

    return {"query": query, "thread_id": thread_id}


# ---------------------------------------------------------------------------
# Response collector
# ---------------------------------------------------------------------------

def _collect(
    reply_queue: "queue.Queue[MessageEnvelope]",
    expected: dict[str, str],   # request_id → intent
    timeout: float,
) -> dict[str, MessageEnvelope]:
    """Block until all expected request_ids arrive or timeout expires.

    Returns a dict of request_id → MessageEnvelope.
    Missing entries indicate timeout; the caller checks len(result) < len(expected).
    """
    responses: dict[str, MessageEnvelope] = {}
    deadline = time.monotonic() + timeout
    while len(responses) < len(expected):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            resp = reply_queue.get(timeout=remaining)
            responses[resp.request_id] = resp
        except queue.Empty:
            break
    return responses


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class Dispatcher:
    """Routes chat requests to Worker pools via staged IntentGraph execution.

    Replaces build_minimal_supervisor_graph / run_minimal_graph.
    Keeps Mem0 memory search and InputSafetyFilter logic intact.
    """

    def __init__(self, team: AgentTeam, mem0: Mem0Store | None = None) -> None:
        self._team = team
        self._mem0 = mem0

    def dispatch(
        self,
        query: str,
        *,
        book_id: str | None = None,
        thread_id: str = "default",
        active_tab: str | None = None,
        selected_text: str | None = None,
        current_chapter: str | None = None,
    ) -> DispatchResult:
        """Main entry point. Called from chat.py. Synchronous; FastAPI runs it
        in its thread pool when the endpoint is declared as `def` (not async def).
        """
        # 1. Safety check (reset every turn — same logic as original safety_ok=None)
        safety = run_input_safety_check(query)
        if not safety.allowed:
            print(f"[Dispatcher] safety blocked: {safety.reason}", file=sys.stdout)
            return DispatchResult(
                answer=f"当前请求未通过安全检查：{safety.reason}",
                blocked=True,
            )

        # 2. Resolve book_id → metadata
        book_meta: dict | None = None
        if book_id:
            book_meta = get_book_catalog().get_by_id(book_id)

        book_title = book_meta.get("title", "") if book_meta else ""

        # 3. Mem0 memory search
        memory_ctx = ""
        if self._mem0:
            try:
                past = self._mem0.search(query, top_k=3)
                if past:
                    memory_ctx = "\n".join(f"- {m}" for m in past)
                    print(f"[Dispatcher] found {len(past)} memory entries", file=sys.stdout)
            except Exception as exc:
                print(f"[Dispatcher] mem0 search failed: {exc}", file=sys.stderr)

        # 4. Intent classification → IntentGraph
        intent_graph = _classify_intent(query, book_title, active_tab)
        print(f"[Dispatcher] intent_graph={intent_graph.stages}", file=sys.stdout)

        # 5. Staged execution
        context: dict[str, dict] = {}   # upstream intent → payload dict
        all_responses: dict[str, MessageEnvelope] = {}
        all_requests: dict[str, str] = {}   # request_id → intent (ordered)
        intent_order: list[str] = []

        for stage in intent_graph.stages:
            reply_queue: queue.Queue[MessageEnvelope] = queue.Queue()
            stage_requests: dict[str, str] = {}

            for intent in stage:
                if intent not in self._team.pools:
                    print(f"[Dispatcher] unknown intent {intent!r}, skipping", file=sys.stderr)
                    continue
                payload = _build_payload(
                    intent, query, book_meta, memory_ctx,
                    thread_id, selected_text, current_chapter, context,
                )
                req_id = uuid.uuid4().hex
                stage_requests[req_id] = intent
                all_requests[req_id] = intent
                if intent not in intent_order:
                    intent_order.append(intent)
                worker = self._team.pools[intent].least_busy()
                print(f"[Dispatcher] → {intent} (worker={worker.name})", file=sys.stdout)
                worker.inbox.put(MessageEnvelope(
                    msg_type="task_request",
                    request_id=req_id,
                    sender="dispatcher",
                    payload=payload,
                    reply_to=reply_queue,
                ))

            if not stage_requests:
                continue

            # Wait for this stage to complete
            stage_responses = _collect(reply_queue, stage_requests, timeout=60.0)
            all_responses.update(stage_responses)

            # Update context with this stage's outputs (for next stage injection)
            for req_id, resp in stage_responses.items():
                intent = stage_requests[req_id]
                if resp.msg_type == "task_response":
                    context[intent] = resp.payload

            # Log timeouts
            missing = set(stage_requests) - set(stage_responses)
            if missing:
                print(
                    f"[Dispatcher] stage timeout, missing request_ids: {missing}",
                    file=sys.stderr,
                )

        # 6. Aggregate
        result = self._aggregate(intent_order, all_requests, all_responses)

        # 7. Save to Mem0
        if self._mem0 and result.answer and not result.blocked:
            try:
                self._mem0.add_qa(query, result.answer)
            except Exception as exc:
                print(f"[Dispatcher] mem0 save failed: {exc}", file=sys.stderr)

        return result

    def _aggregate(
        self,
        intent_order: list[str],
        all_requests: dict[str, str],
        all_responses: dict[str, MessageEnvelope],
    ) -> DispatchResult:
        if not all_requests:
            return DispatchResult(answer="无法识别您的请求，请重试。")

        # Build intent → first response mapping (in execution order)
        intent_to_resp: dict[str, MessageEnvelope] = {}
        for req_id, intent in all_requests.items():
            if req_id in all_responses and intent not in intent_to_resp:
                intent_to_resp[intent] = all_responses[req_id]

        # Single intent
        if len(intent_order) == 1:
            intent = intent_order[0]
            resp = intent_to_resp.get(intent)
            if resp is None:
                return DispatchResult(answer="请求超时，请稍后重试。", intent=intent)
            if resp.msg_type == "task_error":
                return DispatchResult(
                    answer=f"处理出错：{resp.payload.get('error', '未知错误')}",
                    intent=intent,
                )
            p = resp.payload
            return DispatchResult(
                answer=p.get("answer", ""),
                citations=p.get("citations", []),
                retrieved_docs_count=p.get("retrieved_docs_count", 0),
                intent=p.get("intent", intent),
            )

        # Compound: synthesize with LLM
        parts: list[str] = []
        all_citations: list[dict] = []
        total_docs = 0
        for intent in intent_order:
            resp = intent_to_resp.get(intent)
            if resp and resp.msg_type == "task_response":
                parts.append(f"[{intent}]\n{resp.payload.get('answer', '')}")
                all_citations.extend(resp.payload.get("citations", []))
                total_docs += resp.payload.get("retrieved_docs_count", 0)
            else:
                parts.append(f"[{intent}]\n（该部分未完成）")

        merged = self._synthesize(parts)
        return DispatchResult(
            answer=merged,
            citations=all_citations,
            retrieved_docs_count=total_docs,
            intent="+".join(intent_order),
        )

    def _synthesize(self, parts: list[str]) -> str:
        """Merge multiple agent outputs into a single coherent response."""
        combined = "\n\n---\n\n".join(parts)
        llm = get_llm(temperature=0.2)
        try:
            msg = llm.invoke([
                {
                    "role": "system",
                    "content": (
                        "你是智能阅读助手。将以下多个部分的回答整合为一份连贯、完整的Markdown回复，"
                        "保留所有关键信息，避免重复，语言流畅自然。"
                    ),
                },
                {"role": "user", "content": combined},
            ])
            return getattr(msg, "content", str(msg))
        except Exception as exc:
            print(f"[Dispatcher] synthesize failed: {exc}", file=sys.stderr)
            return combined
