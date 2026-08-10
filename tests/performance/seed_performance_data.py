#!/usr/bin/env python3
"""Seed isolated PostgreSQL data for non-AI JMeter scenarios."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from auth.passwords import hash_password  # noqa: E402
from storage.postgres import ensure_postgres_schema, get_postgres_connection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", default="kant.jmeter.reader@example.com")
    parser.add_argument("--password", default="Kant-JMeter-2026!")
    parser.add_argument("--admin-email", default="kant.jmeter.admin@example.com")
    parser.add_argument("--admin-password", default="Kant-JMeter-2026!")
    parser.add_argument("--book-id", default="kant-performance-book")
    args = parser.parse_args()

    ensure_postgres_schema()
    now = datetime.now(timezone.utc)
    conn = get_postgres_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM users WHERE email = %s", (args.email,))
        row = cur.fetchone()
        user_id = row["user_id"] if row else str(uuid.uuid4())
        cur.execute(
            """
            INSERT INTO users (user_id, email, password_hash, role, status, created_at)
            VALUES (%s, %s, %s, 'member', 'active', %s)
            ON CONFLICT (email) DO UPDATE SET
                password_hash = EXCLUDED.password_hash,
                role = 'member',
                status = 'active'
            """,
            (user_id, args.email, hash_password(args.password), now),
        )
        cur.execute("SELECT user_id FROM users WHERE email = %s", (args.admin_email,))
        row = cur.fetchone()
        admin_user_id = row["user_id"] if row else str(uuid.uuid4())
        cur.execute(
            """
            INSERT INTO users (user_id, email, password_hash, role, status, created_at)
            VALUES (%s, %s, %s, 'admin', 'active', %s)
            ON CONFLICT (email) DO UPDATE SET
                password_hash = EXCLUDED.password_hash,
                role = 'admin',
                status = 'active'
            """,
            (
                admin_user_id,
                args.admin_email,
                hash_password(args.admin_password),
                now,
            ),
        )
        cur.execute(
            """
            INSERT INTO books
                (book_id, owner_user_id, title, author, source, total_chunks,
                 added_at, cover_path, status, progress)
            VALUES (%s, %s, %s, %s, %s, %s, %s, '', 'reading', 0.25)
            ON CONFLICT (book_id) DO UPDATE SET
                owner_user_id = EXCLUDED.owner_user_id,
                title = EXCLUDED.title,
                author = EXCLUDED.author,
                source = EXCLUDED.source,
                total_chunks = EXCLUDED.total_chunks,
                status = EXCLUDED.status,
                progress = EXCLUDED.progress
            """,
            (
                args.book_id,
                user_id,
                "Kant Performance Test Book",
                "Performance Fixture",
                "/ebooks/kant-performance-test.epub",
                100,
                now,
            ),
        )
        cur.execute(
            "DELETE FROM conversations WHERE owner_user_id = %s AND book_id = %s",
            (user_id, args.book_id),
        )
        conn.commit()
    finally:
        conn.close()

    print(
        json.dumps(
            {
                "email": args.email,
                "user_id": user_id,
                "admin_email": args.admin_email,
                "admin_user_id": admin_user_id,
                "book_id": args.book_id,
            }
        )
    )


if __name__ == "__main__":
    main()
