from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.deps import require_admin
from storage.audit_log_catalog import AuditLogCatalog
from storage.book_catalog import get_book_catalog
from storage.conversation_catalog import ConversationCatalog
from storage.user_catalog import UserCatalog


router = APIRouter(prefix="/api/admin", tags=["admin"])


def _log_admin_view(request: Request, *, admin_user: dict, action: str, resource_type: str, resource_id: str) -> None:
    AuditLogCatalog().write(
        actor_user_id=admin_user["user_id"],
        actor_role=admin_user["role"],
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        result="ok",
        ip=(request.client.host if request.client else ""),
        user_agent=request.headers.get("user-agent", ""),
    )


@router.get("/users")
def list_users(request: Request, current_admin: dict = Depends(require_admin)) -> list[dict]:
    _log_admin_view(request, admin_user=current_admin, action="admin_list_users", resource_type="user", resource_id="*")
    return UserCatalog().list_all()


@router.get("/users/{user_id}/books")
def list_user_books(user_id: str, request: Request, current_admin: dict = Depends(require_admin)) -> list[dict]:
    _log_admin_view(request, admin_user=current_admin, action="admin_list_books", resource_type="user_books", resource_id=user_id)
    return get_book_catalog().get_all(owner_user_id=user_id)


@router.get("/users/{user_id}/books/{book_id}/conversations")
def list_user_conversations(
    user_id: str,
    book_id: str,
    request: Request,
    current_admin: dict = Depends(require_admin),
) -> list[dict]:
    _log_admin_view(
        request,
        admin_user=current_admin,
        action="admin_list_conversations",
        resource_type="user_book_conversations",
        resource_id=f"{user_id}:{book_id}",
    )
    return ConversationCatalog().list_by_book(owner_user_id=user_id, book_id=book_id)
