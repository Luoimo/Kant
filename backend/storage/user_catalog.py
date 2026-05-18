from __future__ import annotations

from datetime import datetime, timezone
import uuid

from storage.postgres import get_postgres_connection


class UserCatalog:
    def create_member(self, *, email: str, password_hash: str) -> dict:
        user_id = str(uuid.uuid4())
        conn = get_postgres_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO users (user_id, email, password_hash, role, status, created_at)
                VALUES (%s, %s, %s, 'member', 'active', %s)
                RETURNING user_id, email, role, status, created_at
                """,
                (user_id, email, password_hash, datetime.now(timezone.utc)),
            )
            row = cur.fetchone()
            conn.commit()
            return dict(row)
        finally:
            conn.close()

    def get_by_email(self, email: str) -> dict | None:
        conn = get_postgres_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT user_id, email, password_hash, role, status, created_at
                FROM users
                WHERE email = %s
                """,
                (email,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_by_id(self, user_id: str) -> dict | None:
        conn = get_postgres_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT user_id, email, role, status, created_at
                FROM users
                WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_all(self) -> list[dict]:
        conn = get_postgres_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT user_id, email, role, status, created_at
                FROM users
                ORDER BY created_at DESC
                """
            )
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()
