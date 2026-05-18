# PostgreSQL 存储迁移 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `books` / `notes` 业务元数据和 LangGraph 聊天 checkpoint 从本地 SQLite 迁移到 PostgreSQL，并建立清晰的存储抽象边界，同时保持现有 API 行为不变。

**Architecture:** 采用“仓储 + provider”方案。`backend/storage` 新增 PostgreSQL 连接与 checkpoint 服务，`book_catalog` 保留兼容工厂函数但内部改为 PostgreSQL 实现，`deepread_agent` 不再直接持有数据库文件路径、SQLite 连接或 checkpoint 表名，而是通过 checkpoint 服务访问历史和清理线程。

**Tech Stack:** FastAPI, PostgreSQL, `psycopg[binary]`, `langgraph-checkpoint-postgres`, pytest

---

## 文件结构

### 新建文件

- `backend/storage/postgres.py`
  - PostgreSQL DSN 解析、同步连接、异步连接、初始化 SQL
- `backend/storage/checkpoint_store.py`
  - LangGraph PostgreSQL saver provider
  - 聊天历史读取与线程清理服务
- `backend/tests/storage/test_postgres_catalog.py`
  - `books` / `notes` PostgreSQL 仓储测试
- `backend/tests/storage/test_checkpoint_store.py`
  - checkpoint 历史读取、清理线程、sync/async saver 提供测试

### 修改文件

- `backend/config.py`
  - 新增 PostgreSQL 配置
  - 保留旧 SQLite 配置一段时间用于迁移脚本
- `backend/storage/book_catalog.py`
  - 从 SQLite 实现改为 PostgreSQL 实现
  - 保留 `get_book_catalog()` / `get_note_catalog()`
- `backend/agents/deepread_agent.py`
  - 去掉 `sqlite3` / `SqliteSaver` / `AsyncSqliteSaver`
  - 改为使用 `checkpoint_store`
- `backend/main.py`
  - 应用启动时初始化 PostgreSQL 存储 provider
- `backend/requirements.txt`
  - 替换 checkpoint 依赖
- `backend/.env.example`
  - 补充 PostgreSQL 环境变量
- `README.md`
  - 更新存储说明和迁移方式
- `backend/tests/api/test_chat_integration.py`
  - 更新 monkeypatch 点，避免依赖旧 SQLite 细节

### 不改文件

- `backend/api/books.py`
  - API 形状保持不变，只继续调用 catalog 工厂函数
- `backend/rag/chroma/chroma_store.py`
  - Chroma Cloud 保持不动
- `backend/storage/oss_client.py`
  - OSS 保持不动

## 任务拆分

### Task 1: 接入 PostgreSQL 配置与依赖

**Files:**
- Create: `backend/storage/postgres.py`
- Modify: `backend/config.py`
- Modify: `backend/requirements.txt`
- Modify: `backend/.env.example`
- Test: `backend/tests/storage/test_postgres_catalog.py`

- [ ] **Step 1: 编写失败测试，约束 PostgreSQL 配置与 schema 初始化入口**

```python
# backend/tests/storage/test_postgres_catalog.py
from pathlib import Path

from config import Settings


def test_settings_exposes_postgres_dsn():
    settings = Settings(postgres_dsn="postgresql://user:pass@localhost:5432/kant")
    assert settings.postgres_dsn.startswith("postgresql://")


def test_postgres_schema_sql_contains_books_and_notes():
    from storage.postgres import CATALOG_SCHEMA_SQL

    normalized = " ".join(CATALOG_SCHEMA_SQL.split())
    assert "CREATE TABLE IF NOT EXISTS books" in normalized
    assert "CREATE TABLE IF NOT EXISTS notes" in normalized
    assert "REFERENCES books(book_id) ON DELETE CASCADE" in normalized
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/storage/test_postgres_catalog.py -v
```

Expected:

```text
FAIL backend/tests/storage/test_postgres_catalog.py::test_settings_exposes_postgres_dsn
E   ValidationError or TypeError because postgres_dsn/storage.postgres does not exist
```

- [ ] **Step 3: 添加最小配置与 PostgreSQL schema 入口**

```python
# backend/config.py
class Settings(BaseSettings):
    ...
    postgres_dsn: str = ""
    postgres_min_pool_size: int = 1
    postgres_max_pool_size: int = 10
```

```python
# backend/storage/postgres.py
from __future__ import annotations

CATALOG_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS books (
    book_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL UNIQUE,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    added_at TIMESTAMPTZ NOT NULL,
    cover_path TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'unread',
    progress DOUBLE PRECISION NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS notes (
    note_id TEXT PRIMARY KEY,
    book_id TEXT NOT NULL UNIQUE REFERENCES books(book_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);
"""
```

```text
# backend/requirements.txt
langgraph-checkpoint-postgres>=2.0.0
psycopg[binary]>=3.2.0
```

```dotenv
# backend/.env.example
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/kant
POSTGRES_MIN_POOL_SIZE=1
POSTGRES_MAX_POOL_SIZE=10
```

- [ ] **Step 4: 运行测试，确认通过**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/storage/test_postgres_catalog.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 5: 提交**

```bash
git add backend/config.py backend/storage/postgres.py backend/requirements.txt backend/.env.example backend/tests/storage/test_postgres_catalog.py
git commit -m "feat: add postgres config and schema bootstrap"
```

### Task 2: 将 book catalog 和 note catalog 改为 PostgreSQL 仓储

**Files:**
- Modify: `backend/storage/book_catalog.py`
- Modify: `backend/storage/postgres.py`
- Test: `backend/tests/storage/test_postgres_catalog.py`

- [ ] **Step 1: 为 PostgreSQL catalog 行为补充失败测试**

```python
# backend/tests/storage/test_postgres_catalog.py
from datetime import datetime, timezone


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))
        return self

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows=None):
        self.cursor_obj = FakeCursor(rows=rows)
        self.committed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def rollback(self):
        pass


def test_book_catalog_add_uses_postgres_upsert(monkeypatch):
    from storage.book_catalog import BookCatalog

    conn = FakeConnection()
    catalog = BookCatalog(conn)

    catalog.add(
        book_id="book-1",
        title="Test Book",
        author="Author",
        source="oss://bucket/book.epub",
        total_chunks=12,
        cover_path="oss://bucket/cover.png",
    )

    sql, params = conn.cursor_obj.executed[0]
    assert "INSERT INTO books" in sql
    assert "ON CONFLICT (source) DO UPDATE" in sql
    assert params[0] == "book-1"
    assert conn.committed is True


def test_note_catalog_get_by_book_id_returns_dict():
    from storage.book_catalog import NoteCatalog

    row = {
        "note_id": "note-1",
        "book_id": "book-1",
        "file_path": "data/notes/book-1.md",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    conn = FakeConnection(rows=[row])
    catalog = NoteCatalog(conn)

    result = catalog.get_by_book_id("book-1")
    assert result["book_id"] == "book-1"
    assert result["file_path"].endswith("book-1.md")
```

- [ ] **Step 2: 运行测试，确认失败**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/storage/test_postgres_catalog.py -v
```

Expected:

```text
FAIL because BookCatalog currently expects a SQLite db path instead of a PostgreSQL connection
```

- [ ] **Step 3: 实现 PostgreSQL 版本的 catalog**

```python
# backend/storage/book_catalog.py
from __future__ import annotations

import uuid as _uuid
from datetime import datetime, timezone
from typing import Any

from storage.postgres import get_postgres_connection


class _DB:
    def __init__(self, conn: Any | None = None) -> None:
        self._conn = conn

    def _connection(self):
        return self._conn or get_postgres_connection()


class BookCatalog(_DB):
    def add(self, *, book_id: str, title: str, author: str, source: str, total_chunks: int, cover_path: str = "") -> None:
        added_at = datetime.now(tz=timezone.utc)
        conn = self._connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO books (book_id, title, author, source, total_chunks, added_at, cover_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source) DO UPDATE SET
                title = EXCLUDED.title,
                author = EXCLUDED.author,
                total_chunks = EXCLUDED.total_chunks,
                added_at = EXCLUDED.added_at,
                cover_path = EXCLUDED.cover_path
            """,
            (book_id, title, author, source, total_chunks, added_at, cover_path),
        )
        conn.commit()

    def get_by_id(self, book_id: str) -> dict | None:
        conn = self._connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM books WHERE book_id = %s", (book_id,))
        row = cur.fetchone()
        return dict(row) if row else None
```

```python
# backend/storage/postgres.py
from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from config import get_settings


def get_postgres_connection():
    settings = get_settings()
    if not settings.postgres_dsn:
        raise RuntimeError("POSTGRES_DSN is required")
    return psycopg.connect(settings.postgres_dsn, row_factory=dict_row)
```

- [ ] **Step 4: 继续补齐 `update_progress` / `update_status` / `get_all` / `get_by_source` / `delete` 与 `NoteCatalog` 的最小实现并跑测试**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/storage/test_postgres_catalog.py -v
```

Expected:

```text
all tests in backend/tests/storage/test_postgres_catalog.py pass
```

- [ ] **Step 5: 提交**

```bash
git add backend/storage/book_catalog.py backend/storage/postgres.py backend/tests/storage/test_postgres_catalog.py
git commit -m "feat: migrate book catalog to postgres"
```

### Task 3: 引入 PostgreSQL checkpoint provider，并把 DeepReadAgent 从 SQLite 解耦

**Files:**
- Create: `backend/storage/checkpoint_store.py`
- Modify: `backend/agents/deepread_agent.py`
- Modify: `backend/main.py`
- Test: `backend/tests/storage/test_checkpoint_store.py`
- Test: `backend/tests/api/test_chat_integration.py`

- [ ] **Step 1: 编写 checkpoint store 的失败测试**

```python
# backend/tests/storage/test_checkpoint_store.py
from langchain_core.messages import HumanMessage, AIMessage


class FakeTuple:
    def __init__(self, messages):
        self.checkpoint = {"channel_values": {"messages": messages}}


class FakeSaver:
    def __init__(self):
        self.deleted = []

    def get_tuple(self, config):
        return FakeTuple([HumanMessage(content="Q"), AIMessage(content="A")])

    def delete_thread(self, thread_id: str):
        self.deleted.append(thread_id)


def test_load_chat_history_returns_user_and_ai_messages():
    from storage.checkpoint_store import load_chat_history

    saver = FakeSaver()
    history = load_chat_history(saver, thread_id="user-1_book-1")

    assert history == [
        {"role": "user", "content": "Q"},
        {"role": "ai", "content": "A"},
    ]


def test_clear_chat_history_delegates_to_store():
    from storage.checkpoint_store import clear_chat_history

    saver = FakeSaver()
    clear_chat_history(saver, thread_id="user-1_book-1")
    assert saver.deleted == ["user-1_book-1"]
```

- [ ] **Step 2: 运行测试，确认失败**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/storage/test_checkpoint_store.py -v
```

Expected:

```text
FAIL because storage.checkpoint_store does not exist
```

- [ ] **Step 3: 实现 checkpoint store 的最小版本**

```python
# backend/storage/checkpoint_store.py
from __future__ import annotations

from langchain_core.messages import BaseMessage


def build_thread_id(*, user_id: str = "default", book_id: str = "", thread_id: str = "default") -> str:
    return f"{user_id}_{book_id}" if book_id else thread_id


def load_chat_history(saver, *, thread_id: str) -> list[dict]:
    tuple_ = saver.get_tuple({"configurable": {"thread_id": thread_id}})
    if not tuple_:
        return []
    messages = tuple_.checkpoint.get("channel_values", {}).get("messages", [])
    history = []
    for m in messages:
        if not isinstance(m, BaseMessage):
            continue
        if m.type in ["human", "user"]:
            history.append({"role": "user", "content": str(m.content).strip()})
        elif m.type in ["ai", "assistant"] and m.content:
            history.append({"role": "ai", "content": m.content})
    return history


def clear_chat_history(saver, *, thread_id: str) -> None:
    saver.delete_thread(thread_id)
```

- [ ] **Step 4: 给 `deepread_agent.py` 写失败测试，约束它不再直接依赖 SQLite**

```python
# backend/tests/api/test_chat_integration.py
def test_fake_agent_interface_does_not_require_sqlite_details():
    agent = FakeAgent()
    result = agent.run(query="hello", book_id="book-1", user_id="user-1")
    assert result.answer
```

```python
# 额外新增测试片段，放在 backend/tests/storage/test_checkpoint_store.py
def test_build_thread_id_prefers_book_scoped_thread():
    from storage.checkpoint_store import build_thread_id

    assert build_thread_id(user_id="u1", book_id="b1", thread_id="ignored") == "u1_b1"
    assert build_thread_id(user_id="u1", book_id="", thread_id="t1") == "t1"
```

- [ ] **Step 5: 将 `deepread_agent.py` 改为依赖 checkpoint store，并更新启动注入**

```python
# backend/agents/deepread_agent.py
from storage.checkpoint_store import (
    build_thread_id,
    clear_chat_history as clear_checkpoint_thread,
    load_chat_history,
    make_async_postgres_saver,
    make_postgres_saver,
)


actual_thread = build_thread_id(user_id=user_id, book_id=book_id, thread_id=thread_id)
with make_postgres_saver() as memory:
    react_agent, current_docs = self._build(
        book_source=book_source,
        book_id=book_id,
        memory=memory,
        sys_msg=sys_msg,
        locale=locale,
    )
```

```python
# backend/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    from storage.postgres import ensure_postgres_schema

    ensure_postgres_schema()
    ...
```

- [ ] **Step 6: 实现 PostgreSQL saver provider，并运行相关测试**

```python
# backend/storage/checkpoint_store.py
from contextlib import contextmanager, asynccontextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from storage.postgres import get_postgres_connection, get_postgres_async_connection


@contextmanager
def make_postgres_saver():
    with get_postgres_connection() as conn:
        saver = PostgresSaver(conn)
        saver.setup()
        yield saver


@asynccontextmanager
async def make_async_postgres_saver():
    async with await get_postgres_async_connection() as conn:
        saver = AsyncPostgresSaver(conn)
        await saver.setup()
        yield saver
```

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/storage/test_checkpoint_store.py backend/tests/api/test_chat_integration.py -v
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 7: 提交**

```bash
git add backend/storage/checkpoint_store.py backend/agents/deepread_agent.py backend/main.py backend/tests/storage/test_checkpoint_store.py backend/tests/api/test_chat_integration.py
git commit -m "refactor: move chat checkpoints to postgres store"
```

### Task 4: 增加 PostgreSQL 启动初始化和 catalog 集成测试

**Files:**
- Modify: `backend/storage/postgres.py`
- Modify: `backend/main.py`
- Test: `backend/tests/storage/test_postgres_catalog.py`

- [ ] **Step 1: 为 schema 初始化和连接校验补充失败测试**

```python
# backend/tests/storage/test_postgres_catalog.py
def test_ensure_postgres_schema_executes_catalog_schema(monkeypatch):
    from storage import postgres

    executed = []

    class FakeConn:
        def execute(self, sql):
            executed.append(sql)

        def commit(self):
            executed.append("commit")

    monkeypatch.setattr(postgres, "get_postgres_connection", lambda: FakeConn())
    postgres.ensure_postgres_schema()

    assert any("CREATE TABLE IF NOT EXISTS books" in item for item in executed)
    assert "commit" in executed
```

- [ ] **Step 2: 运行测试，确认失败**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/storage/test_postgres_catalog.py -v
```

Expected:

```text
FAIL because ensure_postgres_schema is not implemented
```

- [ ] **Step 3: 实现启动初始化**

```python
# backend/storage/postgres.py
def ensure_postgres_schema() -> None:
    conn = get_postgres_connection()
    try:
        with conn.cursor() as cur:
            for stmt in CATALOG_SCHEMA_SQL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
        conn.commit()
    finally:
        conn.close()
```

```python
# backend/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    from storage.postgres import ensure_postgres_schema

    ensure_postgres_schema()
    ...
```

- [ ] **Step 4: 运行测试，确认通过**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/storage/test_postgres_catalog.py -v
```

Expected:

```text
all tests in backend/tests/storage/test_postgres_catalog.py pass
```

- [ ] **Step 5: 提交**

```bash
git add backend/storage/postgres.py backend/main.py backend/tests/storage/test_postgres_catalog.py
git commit -m "feat: initialize postgres schema on startup"
```

### Task 5: 清理 SQLite 业务依赖并更新最终验证

**Files:**
- Modify: `backend/agents/deepread_agent.py`
- Modify: `backend/storage/book_catalog.py`
- Modify: `backend/requirements.txt`
- Modify: `README.md`
- Test: `backend/tests/api/test_chat_integration.py`
- Test: `backend/tests/storage/test_postgres_catalog.py`
- Test: `backend/tests/storage/test_checkpoint_store.py`

- [ ] **Step 1: 写回归测试，约束业务代码不再引用 SQLite 关键字**

```python
# backend/tests/storage/test_checkpoint_store.py
from pathlib import Path


def test_business_storage_modules_no_longer_reference_sqlite():
    deepread_source = Path("backend/agents/deepread_agent.py").read_text(encoding="utf-8")
    catalog_source = Path("backend/storage/book_catalog.py").read_text(encoding="utf-8")
    requirements = Path("backend/requirements.txt").read_text(encoding="utf-8")

    assert "import sqlite3" not in deepread_source
    assert "SqliteSaver" not in deepread_source
    assert "import sqlite3" not in catalog_source
    assert "langgraph-checkpoint-sqlite" not in requirements
```

- [ ] **Step 2: 运行测试，确认失败**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/storage/test_checkpoint_store.py -v
```

Expected:

```text
FAIL until all SQLite-specific imports and requirements are removed
```

- [ ] **Step 3: 清理最终残留，并补 README 存储说明**

```md
# README.md
- `books` / `notes` 元数据存储在 PostgreSQL
- LangGraph chat checkpoint 存储在 PostgreSQL
- Chroma Cloud 负责向量检索存储，不依赖本地业务 SQLite
```

```text
# backend/requirements.txt
删除:
langgraph-checkpoint-sqlite>=2.0.0
```

- [ ] **Step 4: 跑最终测试集**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/storage/test_postgres_catalog.py backend/tests/storage/test_checkpoint_store.py backend/tests/api/test_chat_integration.py -v
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 5: 提交**

```bash
git add backend/agents/deepread_agent.py backend/storage/book_catalog.py backend/requirements.txt README.md backend/tests/storage/test_checkpoint_store.py backend/tests/storage/test_postgres_catalog.py backend/tests/api/test_chat_integration.py
git commit -m "refactor: remove sqlite business storage dependencies"
```

## 自检

### Spec coverage

- PostgreSQL 配置与依赖：Task 1
- `books` / `notes` PostgreSQL 仓储：Task 2
- LangGraph checkpoint PostgreSQL provider：Task 3
- 应用启动初始化：Task 4
- 清理 SQLite 业务依赖与最终验证：Task 5

无缺口。

### Placeholder scan

- 没有 `TBD` / `TODO`
- 每个任务都有明确文件路径、命令、预期结果
- 所有代码步骤都给了最小代码骨架

### Type consistency

- 配置字段统一使用 `postgres_dsn`
- thread 统一通过 `build_thread_id(...)`
- 历史读取与清理统一通过 `storage.checkpoint_store`

无命名冲突。
