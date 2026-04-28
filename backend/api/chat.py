from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from prompts import get_prompts, normalize_locale

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    user_id: str = "default"
    book_id: str | None = None
    thread_id: str = "default"
    active_tab: str | None = None
    selected_text: str | None = None
    current_chapter: str | None = None
    locale: str | None = None   # "zh-CN" or "en-US"; falls back to zh-CN server-side


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    retrieved_docs_count: int
    intent: str | None
    followups: list[str] = []


def _tool_status_label(tool_name: str, locale: str | None) -> str:
    p = get_prompts(locale).router
    return {
        "search_book_content": p.tool_status_book,
        "search_past_notes": p.tool_status_notes,
    }.get(tool_name, p.tool_status_default)


def _safety_denied_text(reason: str, locale: str | None) -> str:
    if normalize_locale(locale) == "en-US":
        return f"This request did not pass the safety check: {reason}"
    return f"当前请求未通过安全检查：{reason}"


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


@router.delete("/chat/history")
def delete_chat_history(
    book_id: str | None = None,
    user_id: str = "default",
    thread_id: str = "default",
    request: Request = None
) -> dict[str, str]:
    agent = request.app.state.agent
    agent.clear_chat_history(book_id=book_id or "", user_id=user_id, thread_id=thread_id)
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request, bg: BackgroundTasks) -> ChatResponse:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=422, detail="query 不能为空")

    locale = normalize_locale(req.locale)

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
            answer=_safety_denied_text(safety.reason, locale),
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
        locale=locale,
    )

    if book_title:
        bg.add_task(_run_note_hook, note_agent, req.query, result.answer, book_title, book_id, locale)

    if mem0 and result.answer:
        try:
            mem0.add_qa(req.query, result.answer)
        except Exception as exc:
            print(f"[chat] mem0 save failed: {exc}")

    followups = []
    if followup_agent and result.answer:
        followups = followup_agent.generate(req.query, result.answer, locale=locale)

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

    locale = normalize_locale(req.locale)
    router_prompts = get_prompts(locale).router

    agent = request.app.state.agent
    mem0 = request.app.state.mem0
    note_agent = request.app.state.note_agent
    followup_agent = getattr(request.app.state, "followup_agent", None)
    router_agent = getattr(request.app.state, "router_agent", None)
    critic_agent = getattr(request.app.state, "critic_agent", None)

    from security.input_filter import run_lakera_guard_check
    safety = run_lakera_guard_check(req.query)
    if not safety.allowed:
        denied_msg = _safety_denied_text(safety.reason, locale)
        async def _denied():
            yield f"data: {json.dumps({'type': 'error', 'message': denied_msg}, ensure_ascii=False)}\n\n"
        return StreamingResponse(_denied(), media_type="text/event-stream")

    book_id = req.book_id or ""
    book_source, book_title = _resolve_book(book_id)
    memory_ctx = _fetch_memory(mem0, req.query)
    query = _build_query(req, book_title)

    async def event_generator():
        answer_parts: list[str] = []
        docs_count = 0
        docs_text = ""

        # 1. Routing
        optimized_query = query
        if router_agent:
            yield f"data: {json.dumps({'type': 'status', 'text': router_prompts.intent_status}, ensure_ascii=False)}\n\n"
            route_info = await router_agent.aroute(req.query, locale=locale)
            if route_info.get("intent") == "book_qa":
                optimized_query = _build_query(
                    ChatRequest(**{**req.model_dump(), "query": route_info["optimized_query"]}),
                    book_title,
                )

        # Signal immediately so the frontend can drop the typing indicator
        yield f"data: {json.dumps({'type': 'thinking'}, ensure_ascii=False)}\n\n"

        # 2. Main answering (DeepReadAgent)
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
                locale=locale,
            ):
                if event_type == "token":
                    answer_parts.append(data)
                    yield f"data: {json.dumps({'type': 'token', 'text': data}, ensure_ascii=False)}\n\n"
                elif event_type == "tool":
                    label = _tool_status_label(data, locale)
                    yield f"data: {json.dumps({'type': 'status', 'text': label}, ensure_ascii=False)}\n\n"
                elif event_type == "done":
                    docs_count = data["docs_count"]
                    docs_text = "\n".join([c.get("snippet", "") for c in data.get("citations", [])])
                    yield f"data: {json.dumps({'type': 'done', 'citations': data['citations'], 'docs_count': docs_count}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
            return

        full_answer = "".join(answer_parts)

        # 3. Critic review — streamed to the user
        critic_full = ""
        if critic_agent and full_answer:
            try:
                async for critic_chunk in critic_agent.aevaluate(
                    req.query, docs_text, full_answer, locale=locale,
                ):
                    critic_full += critic_chunk
                    yield f"data: {json.dumps({'type': 'token', 'text': critic_chunk}, ensure_ascii=False)}\n\n"
            except Exception as e:
                print(f"[chat] critic failed: {e}")

        # Post-stream side effects — fire and forget
        tasks: list = []

        if critic_full:
            tasks.append(asyncio.create_task(
                asyncio.to_thread(
                    agent.add_ai_message,
                    content=critic_full,
                    book_id=book_id,
                    user_id=req.user_id,
                    thread_id=req.thread_id,
                    locale=locale,
                )
            ))

        if book_title:
            tasks.append(asyncio.create_task(
                asyncio.to_thread(
                    _run_note_hook, note_agent, req.query, full_answer, book_title, book_id, locale,
                )
            ))
        if mem0 and full_answer:
            tasks.append(asyncio.create_task(
                asyncio.to_thread(mem0.add_qa, req.query, full_answer)
            ))

        if followup_agent and full_answer:
            followups = await followup_agent.agenerate(req.query, full_answer, locale=locale)
            if followups:
                yield f"data: {json.dumps({'type': 'followup', 'questions': followups}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _run_note_hook(
    note_agent, query: str, answer: str, book_title: str, book_id: str,
    locale: str | None = None,
) -> None:
    try:
        note_agent.process_qa(query, answer, book_title, book_id=book_id, locale=locale)
    except Exception as exc:
        print(f"[chat] note hook failed: {exc}")
