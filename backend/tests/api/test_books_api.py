from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.books import router as books_router
from api.deps import require_member


class FakeBookCatalog:
    def get_all(self, *, owner_user_id: str) -> list[dict]:
        assert owner_user_id == "user-1"
        return [
            {
                "book_id": "book-1",
                "title": "Kant",
                "author": "Immanuel Kant",
                "source": "data/books/kant.epub",
                "total_chunks": 12,
                "added_at": datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
                "cover_path": "",
                "status": "completed",
                "progress": 1.0,
            }
        ]


def test_list_books_accepts_datetime_added_at(monkeypatch):
    import api.books as books_module

    monkeypatch.setattr(books_module, "get_book_catalog", lambda: FakeBookCatalog())

    app = FastAPI()
    app.include_router(books_router)
    app.dependency_overrides[require_member] = lambda: {"user_id": "user-1", "role": "member"}
    client = TestClient(app)

    response = client.get("/books")

    assert response.status_code == 200
    assert response.json()[0]["added_at"] == "2026-05-20T12:00:00Z"
