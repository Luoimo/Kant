from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.storage.book_catalog import ConversationStorage

router = APIRouter(tags=["chat"])

_TOOL_STATUS: dict[str, str] = {
    "search_books": "正在检索书库…",
}


class ChatRequest(BaseModel):
    query: str
    user_id: str = "default"
    book_id: str | None = None
    thread_id: str = "default"
    selected_text: str | None = None
    current_chapter: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    retrieved_docs_count: int
    intent: str | None


def _resolve_book(book_id: str) -> tuple[str | None, str]:
    """Return (book_source, book_title) for a given book_id."""
    if not book_id:
        return None, ""
    from backend.storage.book_catalog import get_book_catalog
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


def _resolve_thread_id(thread_id: str, user_id: str, book_id: str) -> str:
    """Derive a stable thread ID: use explicit value if provided, else per-user-per-book."""
    if thread_id and thread_id != "default":
        return thread_id
    return f"{user_id}:{book_id or 'global'}"


def _build_query(req: ChatRequest, book_title: str) -> str:
    query = req.query
    if book_title:
        query = f"【当前阅读书籍】：《{book_title}》\n\n{query}"
    if req.selected_text:
        query = f"【用户划选的原文片段】：\n{req.selected_text}\n\n【用户问题】：\n{query}"
    if req.current_chapter:
        query += f"\n\n【当前阅读章节】：{req.current_chapter}"
    return query


@router.get("/chat/history")
def get_chat_history(
    request: Request,
    thread_id: str = "default",
    book_id: str = "",
    user_id: str = "default",
    limit: int = 50,
) -> list[dict]:
    """返回指定 thread 的历史消息，按时间正序。"""
    resolved = _resolve_thread_id(thread_id, user_id, book_id)
    return request.app.state.conv.get_history(resolved, limit=limit)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request, bg: BackgroundTasks) -> ChatResponse:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=422, detail="query 不能为空")

    agent = request.app.state.agent
    mem0 = request.app.state.mem0
    note_agent = request.app.state.note_agent

    from backend.security.input_filter import run_input_safety_check
    safety = run_input_safety_check(req.query)
    if not safety.allowed:
        return ChatResponse(
            answer=f"当前请求未通过安全检查：{safety.reason}",
            citations=[],
            retrieved_docs_count=0,
            intent=None,
        )

    book_id = req.book_id or ""
    book_source, book_title = _resolve_book(book_id)
    memory_ctx = _fetch_memory(mem0, req.query)
    query = _build_query(req, book_title)

    conv: ConversationStorage = request.app.state.conv
    thread_id = _resolve_thread_id(req.thread_id, req.user_id, book_id)
    history = conv.get_history(thread_id)

    result = agent.run(
        query=query,
        book_source=book_source,
        book_id=book_id,
        memory_context=memory_ctx,
        user_id=req.user_id,
        history=history,
    )

    if book_title and result.retrieved_docs:
        bg.add_task(_run_note_hook, note_agent, req.query, result.answer, book_title, book_id)

    if mem0 and result.answer:
        try:
            mem0.add_qa(req.query, result.answer)
        except Exception as exc:
            print(f"[chat] mem0 save failed: {exc}")

    if result.answer:
        _save_conversation(
            conv, thread_id, book_id, req.user_id, req.query, result.answer,
            citations=[c.__dict__ for c in result.citations],
        )

    return ChatResponse(
        answer=result.answer,
        citations=[c.__dict__ for c in result.citations],
        retrieved_docs_count=len(result.retrieved_docs),
        intent="deepread",
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=422, detail="query 不能为空")

    agent = request.app.state.agent
    mem0 = request.app.state.mem0
    note_agent = request.app.state.note_agent

    from backend.security.input_filter import run_input_safety_check
    safety = run_input_safety_check(req.query)
    if not safety.allowed:
        async def _denied():
            yield f"data: {json.dumps({'type': 'error', 'message': safety.reason})}\n\n"
        return StreamingResponse(_denied(), media_type="text/event-stream")

    book_id = req.book_id or ""
    book_source, book_title = _resolve_book(book_id)
    memory_ctx = _fetch_memory(mem0, req.query)
    query = _build_query(req, book_title)

    conv: ConversationStorage = request.app.state.conv
    thread_id = _resolve_thread_id(req.thread_id, req.user_id, book_id)

    async def event_generator():
        answer_parts: list[str] = []
        docs_count = 0
        stream_citations: list[dict] = []
        # Signal immediately so the frontend can drop the typing indicator
        yield f"data: {json.dumps({'type': 'thinking'}, ensure_ascii=False)}\n\n"
        # Load history after emitting thinking so StreamingResponse headers are already sent
        history = await asyncio.to_thread(conv.get_history, thread_id)
        try:
            async for event_type, data in agent.astream_events(
                query=query,
                book_source=book_source,
                book_id=book_id,
                memory_context=memory_ctx,
                user_id=req.user_id,
                history=history,
            ):
                if event_type == "token":
                    answer_parts.append(data)
                    yield f"data: {json.dumps({'type': 'token', 'text': data}, ensure_ascii=False)}\n\n"
                elif event_type == "tool":
                    label = _TOOL_STATUS.get(data, "正在处理…")
                    yield f"data: {json.dumps({'type': 'status', 'text': label}, ensure_ascii=False)}\n\n"
                elif event_type == "done":
                    docs_count = data["docs_count"]
                    stream_citations = data["citations"]
                    yield f"data: {json.dumps({'type': 'done', 'citations': stream_citations, 'docs_count': docs_count}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
            return

        # Post-stream side effects — fire and forget
        full_answer = "".join(answer_parts)
        tasks: list = []
        if book_title and docs_count > 0:
            tasks.append(asyncio.to_thread(
                _run_note_hook, note_agent, req.query, full_answer, book_title, book_id
            ))
        if mem0 and full_answer:
            tasks.append(asyncio.to_thread(mem0.add_qa, req.query, full_answer))
        if full_answer:
            tasks.append(asyncio.to_thread(
                _save_conversation, conv, thread_id, book_id, req.user_id, req.query, full_answer,
                stream_citations,
            ))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _save_conversation(
    conv: ConversationStorage,
    thread_id: str,
    book_id: str,
    user_id: str,
    query: str,
    answer: str,
    citations: list[dict] | None = None,
) -> None:
    try:
        conv.save_turn(
            thread_id=thread_id, book_id=book_id, user_id=user_id,
            query=query, answer=answer, citations=citations,
        )
    except Exception as exc:
        print(f"[chat] conversation save failed: {exc}")


def _run_note_hook(
    note_agent, query: str, answer: str, book_title: str, book_id: str
) -> None:
    try:
        note_agent.process_qa(query, answer, book_title, book_id=book_id)
    except Exception as exc:
        print(f"[chat] note hook failed: {exc}")
