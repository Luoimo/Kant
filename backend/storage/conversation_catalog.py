from __future__ import annotations

from datetime import datetime, timezone
import uuid

from storage.postgres import get_postgres_connection


class ConversationCatalog:
    def create(self, *, owner_user_id: str, book_id: str, title: str = "") -> dict:
        now = datetime.now(timezone.utc)
        conversation_id = str(uuid.uuid4())
        conn = get_postgres_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO conversations (conversation_id, owner_user_id, book_id, title, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING conversation_id, owner_user_id, book_id, title, created_at, updated_at
                """,
                (conversation_id, owner_user_id, book_id, title, now, now),
            )
            row = cur.fetchone()
            conn.commit()
            return dict(row)
        finally:
            conn.close()

    def list_by_book(self, *, owner_user_id: str, book_id: str) -> list[dict]:
        conn = get_postgres_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT conversation_id, owner_user_id, book_id, title, created_at, updated_at
                FROM conversations
                WHERE owner_user_id = %s AND book_id = %s
                ORDER BY updated_at DESC
                """,
                (owner_user_id, book_id),
            )
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def get(self, *, owner_user_id: str, conversation_id: str) -> dict | None:
        conn = get_postgres_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT conversation_id, owner_user_id, book_id, title, created_at, updated_at
                FROM conversations
                WHERE owner_user_id = %s AND conversation_id = %s
                """,
                (owner_user_id, conversation_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
