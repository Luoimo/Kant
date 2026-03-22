from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from backend.agents.orchestrator_agent import run_minimal_graph
from backend.xai.citation import Citation

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    book_id: str | None = None
    thread_id: str = "default"
    selected_text: str | None = None
    current_chapter: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    retrieved_docs_count: int
    intent: str | None


def _citation_to_dict(c: Any) -> dict[str, Any]:
    if isinstance(c, Citation):
        return asdict(c)
    if isinstance(c, dict):
        return c
    return {"value": str(c)}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    state = run_minimal_graph(
        req.query,
        book_id=req.book_id,
        thread_id=req.thread_id,
        selected_text=req.selected_text,
        current_chapter=req.current_chapter,
    )
    return ChatResponse(
        answer=state.get("answer", ""),
        citations=[_citation_to_dict(c) for c in state.get("citations", [])],
        retrieved_docs_count=state.get("retrieved_docs_count", 0),
        intent=state.get("intent"),
    )
