from __future__ import annotations

from datetime import datetime, timezone
import uuid

from storage.postgres import get_postgres_connection


class AuditLogCatalog:
    def write(
        self,
        *,
        actor_user_id: str,
        actor_role: str,
        action: str,
        resource_type: str,
        resource_id: str,
        result: str,
        ip: str = "",
        user_agent: str = "",
    ) -> None:
        conn = get_postgres_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO audit_logs
                (log_id, actor_user_id, actor_role, action, resource_type, resource_id, result, ip, user_agent, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(uuid.uuid4()),
                    actor_user_id,
                    actor_role,
                    action,
                    resource_type,
                    resource_id,
                    result,
                    ip,
                    user_agent,
                    datetime.now(timezone.utc),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def list_recent(self, *, limit: int = 100) -> list[dict]:
        conn = get_postgres_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT log_id, actor_user_id, actor_role, action, resource_type, resource_id, result, ip, user_agent, created_at
                FROM audit_logs
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()
