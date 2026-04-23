from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter(tags=["chat"])

_TOOL_STATUS: dict[str, str] = {
    "search_book_content": "正在检索书籍内容…",
    "search_past_notes": "正在回顾历史笔记…",
}


class ChatRequest(BaseModel):
    query: str
    user_id: str = "default"
    book_id: str | None = None
    thread_id: str = "default"
    active_tab: str | None = None
    selected_text: str | None = None
    current_chapter: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    retrieved_docs_count: int
    intent: str | None
    followups: list[str] = []


def _resolve_book(book_id: str) -> tuple[str | None, str]:
    """Return (book_source, book_title) for a given book_id."""
    if not book_id:
        return None, ""
    from storage.book_catalog import get_book_catalog
    meta = get_book_catalog().get_by_id(book_id)
    if not meta:
        return None, ""
    return meta.get("source"), meta.get("title", "")


def _fetch_memory(mem0, query: str) -> str:
    if not mem0:
        return ""
    try:
        past = mem0.search(query, top_k=3)
        return "\n".join(f"- {m}" for m in past) if past else ""
    except Exception as exc:
        print(f"[chat] mem0 search failed: {exc}")
        return ""


def _build_query(req: ChatRequest, book_title: str) -> str:
    # 彻底弃用前端的系统字符串拼接，返回干净的原始 query。
    # 动态参数将直接传给 Agent，由 Agent 在内部作为状态或 SystemMessage 处理。
    return req.query


@router.get("/chat/history")
def get_chat_history(
    book_id: str | None = None,
    user_id: str = "default",
    thread_id: str = "default",
    request: Request = None
) -> dict[str, Any]:
    agent = request.app.state.agent
    history = agent.get_chat_history(book_id=book_id or "", user_id=user_id, thread_id=thread_id)
    return {"messages": history}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request, bg: BackgroundTasks) -> ChatResponse:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=422, detail="query 不能为空")

    agent = request.app.state.agent
    mem0 = request.app.state.mem0
    note_agent = request.app.state.note_agent
    followup_agent = getattr(request.app.state, "followup_agent", None)
    router_agent = getattr(request.app.state, "router_agent", None)
    critic_agent = getattr(request.app.state, "critic_agent", None)

    from security.input_filter import run_lakera_guard_check
    safety = run_lakera_guard_check(req.query)
    if not safety.allowed:
        return ChatResponse(
            answer=f"当前请求未通过安全检查：{safety.reason}",
            citations=[],
            retrieved_docs_count=0,
            intent=None,
            followups=[],
        )

    book_id = req.book_id or ""
    book_source, book_title = _resolve_book(book_id)
    memory_ctx = _fetch_memory(mem0, req.query)
    query = _build_query(req, book_title)

    result = agent.run(
        query=query,
        book_source=book_source,
        book_id=book_id,
        memory_context=memory_ctx,
        user_id=req.user_id,
        thread_id=req.thread_id,
        selected_text=req.selected_text,
        current_chapter=req.current_chapter,
    )

    if book_title:
        bg.add_task(_run_note_hook, note_agent, req.query, result.answer, book_title, book_id)

    if mem0 and result.answer:
        try:
            mem0.add_qa(req.query, result.answer)
        except Exception as exc:
            print(f"[chat] mem0 save failed: {exc}")

    followups = []
    if followup_agent and result.answer:
        followups = followup_agent.generate(req.query, result.answer)

    return ChatResponse(
        answer=result.answer,
        citations=[c.__dict__ for c in result.citations],
        retrieved_docs_count=len(result.retrieved_docs),
        intent=req.active_tab or "deepread",
        followups=followups,
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=422, detail="query 不能为空")

    agent = request.app.state.agent
    mem0 = request.app.state.mem0
    note_agent = request.app.state.note_agent
    followup_agent = getattr(request.app.state, "followup_agent", None)
    router_agent = getattr(request.app.state, "router_agent", None)
    critic_agent = getattr(request.app.state, "critic_agent", None)

    from security.input_filter import run_lakera_guard_check
    safety = run_lakera_guard_check(req.query)
    if not safety.allowed:
        async def _denied():
            yield f"data: {json.dumps({'type': 'error', 'message': safety.reason})}\n\n"
        return StreamingResponse(_denied(), media_type="text/event-stream")

    book_id = req.book_id or ""
    book_source, book_title = _resolve_book(book_id)
    memory_ctx = _fetch_memory(mem0, req.query)
    query = _build_query(req, book_title)

    async def event_generator():
        answer_parts: list[str] = []
        docs_count = 0
        docs_text = ""
        
        # 1. 预处理路由阶段 (RouterAgent)
        optimized_query = query
        if router_agent:
            yield f"data: {json.dumps({'type': 'status', 'text': '正在识别意图…'}, ensure_ascii=False)}\n\n"
            route_info = await router_agent.aroute(req.query)
            if route_info.get("intent") == "book_qa":
                optimized_query = _build_query(ChatRequest(**{**req.model_dump(), "query": route_info["optimized_query"]}), book_title)
        
        # Signal immediately so the frontend can drop the typing indicator
        yield f"data: {json.dumps({'type': 'thinking'}, ensure_ascii=False)}\n\n"
        
        # 2. 主查询阶段 (DeepReadAgent)
        try:
            async for event_type, data in agent.astream_events(
                query=optimized_query,
                book_source=book_source,
                book_id=book_id,
                memory_context=memory_ctx,
                user_id=req.user_id,
                thread_id=req.thread_id,
                selected_text=req.selected_text,
                current_chapter=req.current_chapter,
            ):
                if event_type == "token":
                    answer_parts.append(data)
                    yield f"data: {json.dumps({'type': 'token', 'text': data}, ensure_ascii=False)}\n\n"
                elif event_type == "tool":
                    label = _TOOL_STATUS.get(data, "正在处理…")
                    yield f"data: {json.dumps({'type': 'status', 'text': label}, ensure_ascii=False)}\n\n"
                elif event_type == "done":
                    docs_count = data["docs_count"]
                    # 将检索到的证据拼起来供后续 Critic 使用
                    docs_text = "\n".join([c.get("snippet", "") for c in data.get("citations", [])])
                    yield f"data: {json.dumps({'type': 'done', 'citations': data['citations'], 'docs_count': docs_count}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
            return

        full_answer = "".join(answer_parts)

        # 3. 后置评估阶段 (CriticAgent) - 直接向用户流式追加评估结果
        critic_full = ""
        if critic_agent and full_answer:
            try:
                async for critic_chunk in critic_agent.aevaluate(req.query, docs_text, full_answer):
                    critic_full += critic_chunk
                    yield f"data: {json.dumps({'type': 'token', 'text': critic_chunk}, ensure_ascii=False)}\n\n"
            except Exception as e:
                print(f"[chat] critic failed: {e}")

        # Post-stream side effects — fire and forget
        # We start the background tasks immediately without awaiting them here
        # so that we can proceed to generate followups without blocking the stream.
        tasks: list = []
        
        # 将审查笔记补充到聊天记录中
        if critic_full:
            tasks.append(asyncio.create_task(
                asyncio.to_thread(
                    agent.add_ai_message,
                    content=critic_full,
                    book_id=book_id,
                    user_id=req.user_id,
                    thread_id=req.thread_id
                )
            ))
            
        if book_title:
            # Removed the docs_count > 0 condition so notes are saved even for casual chat or non-RAG answers
            tasks.append(asyncio.create_task(
                asyncio.to_thread(
                    _run_note_hook, note_agent, req.query, full_answer, book_title, book_id
                )
            ))
        if mem0 and full_answer:
            tasks.append(asyncio.create_task(
                asyncio.to_thread(mem0.add_qa, req.query, full_answer)
            ))
            
        # Instead of awaiting tasks here, we let them run in the background
        # and immediately start generating followups

        if followup_agent and full_answer:
            followups = await followup_agent.agenerate(req.query, full_answer)
            if followups:
                yield f"data: {json.dumps({'type': 'followup', 'questions': followups}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _run_note_hook(
    note_agent, query: str, answer: str, book_title: str, book_id: str
) -> None:
    try:
        note_agent.process_qa(query, answer, book_title, book_id=book_id)
    except Exception as exc:
        print(f"[chat] note hook failed: {exc}")
