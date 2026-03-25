from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    book_id: str | None = None
    thread_id: str = "default"
    active_tab: str | None = None        # bypasses LLM classifier, routes directly
    selected_text: str | None = None
    current_chapter: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    retrieved_docs_count: int
    intent: str | None


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=422, detail="query 不能为空")

    from backend.team.dispatcher import Dispatcher
    dispatcher: Dispatcher = request.app.state.dispatcher

    result = dispatcher.dispatch(
        req.query,
        book_id=req.book_id,
        thread_id=req.thread_id,
        active_tab=req.active_tab,
        selected_text=req.selected_text,
        current_chapter=req.current_chapter,
    )
    return ChatResponse(
        answer=result.answer,
        citations=result.citations,
        retrieved_docs_count=result.retrieved_docs_count,
        intent=result.intent or None,
    )
