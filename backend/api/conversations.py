from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.deps import require_member
from storage.book_catalog import get_book_catalog
from storage.conversation_catalog import ConversationCatalog


router = APIRouter(prefix="/api/user/books", tags=["conversations"])


class CreateConversationRequest(BaseModel):
    title: str = ""


@router.post("/{book_id}/conversations")
def create_conversation(
    book_id: str,
    req: CreateConversationRequest,
    current_user: dict = Depends(require_member),
) -> dict:
    owner = current_user["user_id"]
    if not get_book_catalog().get_by_id(book_id, owner_user_id=owner):
        raise HTTPException(status_code=404, detail="book not found")
    return ConversationCatalog().create(owner_user_id=owner, book_id=book_id, title=req.title.strip())


@router.get("/{book_id}/conversations")
def list_conversations(
    book_id: str,
    current_user: dict = Depends(require_member),
) -> list[dict]:
    owner = current_user["user_id"]
    if not get_book_catalog().get_by_id(book_id, owner_user_id=owner):
        raise HTTPException(status_code=404, detail="book not found")
    return ConversationCatalog().list_by_book(owner_user_id=owner, book_id=book_id)
