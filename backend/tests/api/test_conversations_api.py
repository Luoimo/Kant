from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.conversations import router as conv_router
from api.deps import get_current_user


class FakeBookCatalog:
    def get_by_id(self, book_id: str, *, owner_user_id: str):
        if book_id == "book-1" and owner_user_id == "user-1":
            return {"book_id": "book-1"}
        return None


class FakeConversationCatalog:
    def __init__(self):
        self.rows = []

    def create(self, *, owner_user_id: str, book_id: str, title: str = "") -> dict:
        row = {"conversation_id": "conv-1", "owner_user_id": owner_user_id, "book_id": book_id, "title": title}
        self.rows.append(row)
        return row

    def list_by_book(self, *, owner_user_id: str, book_id: str) -> list[dict]:
        return [r for r in self.rows if r["owner_user_id"] == owner_user_id and r["book_id"] == book_id]


def test_create_and_list_conversations(monkeypatch):
    import api.conversations as conversations_module

    fake_catalog = FakeConversationCatalog()
    monkeypatch.setattr(conversations_module, "get_book_catalog", lambda: FakeBookCatalog())
    monkeypatch.setattr(conversations_module, "ConversationCatalog", lambda: fake_catalog)

    app = FastAPI()
    app.include_router(conv_router)
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "user-1", "role": "member"}
    c = TestClient(app)

    r = c.post("/api/user/books/book-1/conversations", json={"title": "第一轮讨论"})
    assert r.status_code == 200
    assert r.json()["conversation_id"] == "conv-1"

    r2 = c.get("/api/user/books/book-1/conversations")
    assert r2.status_code == 200
    assert len(r2.json()) == 1
