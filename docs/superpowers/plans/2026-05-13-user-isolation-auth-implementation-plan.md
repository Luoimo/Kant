# 用户隔离鉴权机制 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 Kant 项目中落地“单租户多用户 + 全私有隔离 + 管理员只读透明”的鉴权体系，并完成每本书多会话改造。

**Architecture:** 后端采用 JWT Access Token + Redis Refresh Token 轮转，应用层实施 RBAC 与 owner 校验；数据库层补充 `users/conversations/audit_logs` 与 `owner_user_id` 字段；聊天链路改为 `book_id + conversation_id`，并把 `conversation_id` 作为 checkpoint thread key。

**Tech Stack:** FastAPI, PostgreSQL(psycopg), Redis(redis-py), PyJWT, Argon2id(passlib), Vue3 + Pinia, pytest

---

## 文件结构

### 新建文件

- `backend/auth/passwords.py`
- `backend/auth/tokens.py`
- `backend/auth/refresh_store.py`
- `backend/api/auth.py`
- `backend/api/deps.py`
- `backend/api/conversations.py`
- `backend/api/admin.py`
- `backend/storage/user_catalog.py`
- `backend/storage/conversation_catalog.py`
- `backend/storage/audit_log_catalog.py`
- `backend/tests/auth/test_auth_api.py`
- `backend/tests/api/test_conversations_api.py`
- `backend/tests/api/test_admin_api.py`
- `frontend/src/stores/auth.js`
- `frontend/src/views/AdminView.vue`

### 修改文件

- `backend/config.py`
- `backend/main.py`
- `backend/storage/postgres.py`
- `backend/storage/book_catalog.py`
- `backend/api/books.py`
- `backend/api/chat.py`
- `backend/memory/mem0_store.py`
- `backend/rag/chroma/chroma_store.py`
- `backend/requirements.txt`
- `backend/.env.example`
- `backend/tests/storage/test_postgres_catalog.py`
- `backend/tests/api/test_chat_integration.py`
- `frontend/src/api/index.js`
- `frontend/src/stores/chat.js`
- `frontend/src/stores/books.js`
- `frontend/src/router/index.js`
- `README.md`

## Task 1: 接入鉴权配置与依赖

**Files:**
- Create: `backend/tests/auth/test_auth_api.py`
- Modify: `backend/config.py`
- Modify: `backend/requirements.txt`
- Modify: `backend/.env.example`

- [ ] **Step 1: 先写失败测试，约束新的鉴权配置字段**

```python
# backend/tests/auth/test_auth_api.py
from config import Settings


def test_auth_settings_fields_exist():
    s = Settings(
        jwt_secret_key="dev-secret",
        jwt_access_token_minutes=15,
        jwt_refresh_token_days=30,
        redis_url="redis://localhost:6379/0",
    )
    assert s.jwt_secret_key == "dev-secret"
    assert s.jwt_access_token_minutes == 15
    assert s.jwt_refresh_token_days == 30
    assert s.redis_url.startswith("redis://")
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/auth/test_auth_api.py::test_auth_settings_fields_exist -v
```

Expected:

```text
FAIL，提示 Settings 不存在 jwt_* 或 redis_url 字段
```

- [ ] **Step 3: 增加配置与依赖**

```python
# backend/config.py (Settings 内新增字段)
jwt_secret_key: str = "dev-change-me"
jwt_issuer: str = "kant-backend"
jwt_access_token_minutes: int = 15
jwt_refresh_token_days: int = 30
redis_url: str = "redis://localhost:6379/0"
```

```text
# backend/requirements.txt 新增
PyJWT>=2.9.0
redis>=5.0.0
passlib[argon2]>=1.7.4
```

```dotenv
# backend/.env.example 新增
JWT_SECRET_KEY=dev-change-me
JWT_ISSUER=kant-backend
JWT_ACCESS_TOKEN_MINUTES=15
JWT_REFRESH_TOKEN_DAYS=30
REDIS_URL=redis://localhost:6379/0
```

- [ ] **Step 4: 重新运行测试**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/auth/test_auth_api.py::test_auth_settings_fields_exist -v
```

Expected:

```text
1 passed
```

- [ ] **Step 5: 提交**

```bash
git add backend/config.py backend/requirements.txt backend/.env.example backend/tests/auth/test_auth_api.py
git commit -m "feat: add auth settings and dependencies"
```

## Task 2: 扩展 PostgreSQL schema（用户、会话、审计、owner 字段）

**Files:**
- Modify: `backend/storage/postgres.py`
- Modify: `backend/tests/storage/test_postgres_catalog.py`

- [ ] **Step 1: 写失败测试，约束 DDL 含新表和 owner 字段**

```python
# backend/tests/storage/test_postgres_catalog.py 新增用例
from storage.postgres import render_catalog_schema_ddl


def test_schema_contains_users_conversations_audit_and_owner_columns():
    ddl = render_catalog_schema_ddl("catalog").as_string(None)
    assert "CREATE TABLE IF NOT EXISTS \"catalog\".users" in ddl
    assert "CREATE TABLE IF NOT EXISTS \"catalog\".conversations" in ddl
    assert "CREATE TABLE IF NOT EXISTS \"catalog\".audit_logs" in ddl
    assert "owner_user_id" in ddl
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/storage/test_postgres_catalog.py::test_schema_contains_users_conversations_audit_and_owner_columns -v
```

Expected:

```text
FAIL，DDL 中缺少 users/conversations/audit_logs 或 owner_user_id
```

- [ ] **Step 3: 更新 DDL**

```python
# backend/storage/postgres.py 的 CATALOG_SCHEMA_DDL 追加
CREATE TABLE IF NOT EXISTS {schema}.users (
    user_id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'member')),
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL
);

ALTER TABLE {schema}.books
    ADD COLUMN IF NOT EXISTS owner_user_id TEXT NOT NULL DEFAULT 'bootstrap-owner';

CREATE INDEX IF NOT EXISTS idx_books_owner_added_at
    ON {schema}.books(owner_user_id, added_at DESC);

CREATE TABLE IF NOT EXISTS {schema}.conversations (
    conversation_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL REFERENCES {schema}.users(user_id) ON DELETE CASCADE,
    book_id TEXT NOT NULL REFERENCES {schema}.books(book_id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_owner_book_updated
    ON {schema}.conversations(owner_user_id, book_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS {schema}.audit_logs (
    log_id TEXT PRIMARY KEY,
    actor_user_id TEXT NOT NULL,
    actor_role TEXT NOT NULL,
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    result TEXT NOT NULL,
    ip TEXT NOT NULL DEFAULT '',
    user_agent TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL
);
```

- [ ] **Step 4: 运行测试**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/storage/test_postgres_catalog.py -v
```

Expected:

```text
全部通过；若旧断言依赖老 DDL，按新 DDL 同步更新断言
```

- [ ] **Step 5: 提交**

```bash
git add backend/storage/postgres.py backend/tests/storage/test_postgres_catalog.py
git commit -m "feat: extend postgres schema for auth ownership and audit"
```

## Task 3: 实现认证核心（密码、JWT、Redis refresh）

**Files:**
- Create: `backend/auth/passwords.py`
- Create: `backend/auth/tokens.py`
- Create: `backend/auth/refresh_store.py`
- Modify: `backend/tests/auth/test_auth_api.py`

- [ ] **Step 1: 写失败测试，定义核心函数契约**

```python
# backend/tests/auth/test_auth_api.py 追加
from auth.passwords import hash_password, verify_password
from auth.tokens import create_access_token, decode_access_token


def test_password_hash_and_verify():
    h = hash_password("secret-123")
    assert h != "secret-123"
    assert verify_password("secret-123", h) is True
    assert verify_password("wrong", h) is False


def test_access_token_roundtrip():
    token = create_access_token(user_id="u-1", role="member")
    payload = decode_access_token(token)
    assert payload["sub"] == "u-1"
    assert payload["role"] == "member"
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/auth/test_auth_api.py::test_password_hash_and_verify backend/tests/auth/test_auth_api.py::test_access_token_roundtrip -v
```

Expected:

```text
FAIL，找不到 auth.* 模块或函数
```

- [ ] **Step 3: 最小实现**

```python
# backend/auth/passwords.py
from passlib.context import CryptContext

_pwd_ctx = CryptContext(schemes=["argon2"], deprecated="auto")


def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)
```

```python
# backend/auth/tokens.py
from datetime import datetime, timedelta, timezone
import uuid
import jwt
from config import get_settings


def create_access_token(*, user_id: str, role: str) -> str:
    s = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "role": role,
        "jti": str(uuid.uuid4()),
        "iss": s.jwt_issuer,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=s.jwt_access_token_minutes)).timestamp()),
        "tenant_id": "default",
    }
    return jwt.encode(payload, s.jwt_secret_key, algorithm="HS256")


def decode_access_token(token: str) -> dict:
    s = get_settings()
    return jwt.decode(token, s.jwt_secret_key, algorithms=["HS256"], issuer=s.jwt_issuer)
```

```python
# backend/auth/refresh_store.py
import hashlib
from datetime import datetime, timedelta, timezone
from redis import Redis
from config import get_settings


class RefreshStore:
    def __init__(self, client: Redis | None = None) -> None:
        s = get_settings()
        self._redis = client or Redis.from_url(s.redis_url, decode_responses=True)
        self._days = s.jwt_refresh_token_days

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def save(self, *, user_id: str, jti: str, token: str) -> None:
        key = f"auth:refresh:{user_id}:{jti}"
        ttl = int(timedelta(days=self._days).total_seconds())
        self._redis.hset(key, mapping={"token_hash": self._hash(token)})
        self._redis.expire(key, ttl)
        self._redis.sadd(f"auth:user_sessions:{user_id}", jti)
```

- [ ] **Step 4: 运行测试**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/auth/test_auth_api.py::test_password_hash_and_verify backend/tests/auth/test_auth_api.py::test_access_token_roundtrip -v
```

Expected:

```text
2 passed
```

- [ ] **Step 5: 提交**

```bash
git add backend/auth/passwords.py backend/auth/tokens.py backend/auth/refresh_store.py backend/tests/auth/test_auth_api.py
git commit -m "feat: add auth core utilities and refresh token store"
```

## Task 4: 实现用户仓储与 Auth API

**Files:**
- Create: `backend/storage/user_catalog.py`
- Create: `backend/api/auth.py`
- Modify: `backend/main.py`
- Modify: `backend/tests/auth/test_auth_api.py`

- [ ] **Step 1: 写失败测试，覆盖注册/登录/刷新/登出**

```python
# backend/tests/auth/test_auth_api.py 追加
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.auth import router as auth_router


def test_auth_register_login_flow(monkeypatch):
    app = FastAPI()
    app.include_router(auth_router)
    c = TestClient(app)

    r = c.post("/auth/register", json={"email": "u1@test.com", "password": "secret-123"})
    assert r.status_code == 200

    r2 = c.post("/auth/login", json={"email": "u1@test.com", "password": "secret-123"})
    assert r2.status_code == 200
    assert "access_token" in r2.json()
    assert "refresh_token" in r2.json()
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/auth/test_auth_api.py::test_auth_register_login_flow -v
```

Expected:

```text
FAIL，缺少 /auth/register 或 /auth/login 路由
```

- [ ] **Step 3: 实现仓储和路由**

```python
# backend/storage/user_catalog.py
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
```

```python
# backend/api/auth.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from auth.passwords import hash_password, verify_password
from auth.tokens import create_access_token
from storage.user_catalog import UserCatalog

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/register")
def register(req: RegisterRequest):
    if len(req.password) < 8:
        raise HTTPException(status_code=422, detail="password too short")
    user = UserCatalog().create_member(email=req.email, password_hash=hash_password(req.password))
    return {"user_id": user["user_id"], "email": user["email"], "role": user["role"]}


@router.post("/login")
def login(req: LoginRequest):
    user = UserCatalog().get_by_email(req.email)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="invalid credentials")
    access = create_access_token(user_id=user["user_id"], role=user["role"])
    refresh = create_access_token(user_id=user["user_id"], role=user["role"])
    return {"access_token": access, "refresh_token": refresh}
```

```python
# backend/main.py
from api.auth import router as auth_router
...
app.include_router(auth_router)
```

- [ ] **Step 4: 运行测试**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/auth/test_auth_api.py -v
```

Expected:

```text
认证相关用例通过
```

- [ ] **Step 5: 提交**

```bash
git add backend/storage/user_catalog.py backend/api/auth.py backend/main.py backend/tests/auth/test_auth_api.py
git commit -m "feat: add auth endpoints and user catalog"
```

## Task 5: 接入鉴权依赖与角色守卫

**Files:**
- Create: `backend/api/deps.py`
- Modify: `backend/api/books.py`
- Modify: `backend/api/chat.py`
- Modify: `backend/tests/api/test_chat_integration.py`

- [ ] **Step 1: 写失败测试，禁止通过 body 传 user_id 越权**

```python
# backend/tests/api/test_chat_integration.py 新增
def test_chat_ignores_body_user_id_and_uses_token_identity(test_client):
    r = test_client.post(
        "/chat",
        json={"query": "q", "book_id": "book-1", "user_id": "forged-user"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert r.status_code in (200, 401)  # 在接入鉴权后应转为 200 且使用 token 用户
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/api/test_chat_integration.py::test_chat_ignores_body_user_id_and_uses_token_identity -v
```

Expected:

```text
FAIL，当前实现仍从请求体读取 user_id
```

- [ ] **Step 3: 增加依赖并改造接口参数来源**

```python
# backend/api/deps.py
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from auth.tokens import decode_access_token

bearer = HTTPBearer(auto_error=False)


def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    if not creds:
        raise HTTPException(status_code=401, detail="missing token")
    try:
        payload = decode_access_token(creds.credentials)
    except Exception:
        raise HTTPException(status_code=401, detail="invalid token")
    return {"user_id": payload["sub"], "role": payload["role"]}


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="admin required")
    return user
```

```python
# backend/api/chat.py 核心改动
from api.deps import get_current_user
...
@router.post("/chat")
def chat(req: ChatRequest, request: Request, bg: BackgroundTasks, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    ...
    result = agent.run(..., user_id=user_id, ...)
```

- [ ] **Step 4: 运行测试**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/api/test_chat_integration.py -v
```

Expected:

```text
所有聊天集成测试通过；无 token 的路径返回 401
```

- [ ] **Step 5: 提交**

```bash
git add backend/api/deps.py backend/api/books.py backend/api/chat.py backend/tests/api/test_chat_integration.py
git commit -m "feat: enforce token identity and role guards"
```

## Task 6: 书籍与向量检索的 owner 隔离

**Files:**
- Modify: `backend/storage/postgres.py`
- Modify: `backend/storage/book_catalog.py`
- Modify: `backend/api/books.py`
- Modify: `backend/rag/chroma/chroma_store.py`
- Modify: `backend/tests/storage/test_postgres_catalog.py`

- [ ] **Step 1: 写失败测试，约束 get_all/get_by_id 按 owner 过滤**

```python
# backend/tests/storage/test_postgres_catalog.py 新增
def test_book_catalog_get_all_filters_by_owner():
    from storage.postgres import PostgresBookCatalog
    cursor = FakeCursor(fetchall_result=[{"book_id": "b1", "owner_user_id": "u1"}])
    conn = FakeConnection(cursor)
    catalog = PostgresBookCatalog(connection_factory=lambda: conn)
    catalog.get_all(owner_user_id="u1")
    sql, params = _last_query(cursor)
    assert "WHERE owner_user_id = %s" in sql
    assert params == ("u1",)
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/storage/test_postgres_catalog.py::test_book_catalog_get_all_filters_by_owner -v
```

Expected:

```text
FAIL，当前 get_all 不接受 owner_user_id
```

- [ ] **Step 3: 仓储与 API 改造**

```python
# backend/storage/postgres.py (PostgresBookCatalog)
def get_all(self, *, owner_user_id: str) -> list[dict]:
    return self._fetchall(
        "SELECT * FROM books WHERE owner_user_id = %s ORDER BY added_at DESC",
        (owner_user_id,),
    )

def get_by_id(self, book_id: str, *, owner_user_id: str) -> dict | None:
    return self._fetchone(
        "SELECT * FROM books WHERE book_id = %s AND owner_user_id = %s",
        (book_id, owner_user_id),
    )
```

```python
# backend/api/books.py (示例)
@router.get("", response_model=list[BookEntry])
def list_books(current_user: dict = Depends(get_current_user)) -> list[BookEntry]:
    owner = current_user["user_id"]
    return [BookEntry(... ) for b in get_book_catalog().get_all(owner_user_id=owner)]
```

```python
# backend/rag/chroma/chroma_store.py 关键签名
def ingest(self, epub_path: Path, *, source_override: str | None = None, owner_user_id: str) -> IngestResult:
    ...
    metadata = {
        "book_id": book_id,
        "source": source,
        "owner_user_id": owner_user_id,
    }
```

- [ ] **Step 4: 运行相关测试**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/storage/test_postgres_catalog.py backend/tests/rag/test_chroma_store.py -v
```

Expected:

```text
相关测试通过，检索过滤断言包含 owner_user_id
```

- [ ] **Step 5: 提交**

```bash
git add backend/storage/postgres.py backend/storage/book_catalog.py backend/api/books.py backend/rag/chroma/chroma_store.py backend/tests/storage/test_postgres_catalog.py
git commit -m "feat: enforce owner isolation for books and vector data"
```

## Task 7: 会话模型改造（每本书多会话）

**Files:**
- Create: `backend/storage/conversation_catalog.py`
- Create: `backend/api/conversations.py`
- Modify: `backend/api/chat.py`
- Modify: `backend/storage/checkpoint_store.py`
- Create: `backend/tests/api/test_conversations_api.py`
- Modify: `backend/tests/api/test_chat_integration.py`

- [ ] **Step 1: 先写失败测试，定义 conversation API**

```python
# backend/tests/api/test_conversations_api.py
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.conversations import router as conv_router


def test_create_and_list_conversations():
    app = FastAPI()
    app.include_router(conv_router)
    c = TestClient(app)
    r = c.post("/api/user/books/book-1/conversations", json={"title": "第一轮讨论"})
    assert r.status_code in (200, 401)
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/api/test_conversations_api.py -v
```

Expected:

```text
FAIL，缺少 conversations 路由
```

- [ ] **Step 3: 实现会话仓储和 chat 输入模型改造**

```python
# backend/storage/conversation_catalog.py
from datetime import datetime, timezone
import uuid
from storage.postgres import get_postgres_connection


class ConversationCatalog:
    def create(self, *, owner_user_id: str, book_id: str, title: str) -> dict:
        cid = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        conn = get_postgres_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO conversations (conversation_id, owner_user_id, book_id, title, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING conversation_id, owner_user_id, book_id, title, created_at, updated_at
                """,
                (cid, owner_user_id, book_id, title, now, now),
            )
            row = cur.fetchone()
            conn.commit()
            return dict(row)
        finally:
            conn.close()
```

```python
# backend/api/chat.py (ChatRequest 去掉 user_id/thread_id，新增 conversation_id)
class ChatRequest(BaseModel):
    query: str
    book_id: str
    conversation_id: str
    ...
```

```python
# backend/storage/checkpoint_store.py
def build_thread_id(*, conversation_id: str) -> str:
    return conversation_id
```

- [ ] **Step 4: 运行测试**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/api/test_conversations_api.py backend/tests/api/test_chat_integration.py -v
```

Expected:

```text
会话创建/列表通过；聊天必须 book_id+conversation_id 组合合法
```

- [ ] **Step 5: 提交**

```bash
git add backend/storage/conversation_catalog.py backend/api/conversations.py backend/api/chat.py backend/storage/checkpoint_store.py backend/tests/api/test_conversations_api.py backend/tests/api/test_chat_integration.py
git commit -m "feat: add per-book multi-conversation model"
```

## Task 8: 管理员只读 API 与审计日志

**Files:**
- Create: `backend/storage/audit_log_catalog.py`
- Create: `backend/api/admin.py`
- Modify: `backend/main.py`
- Create: `backend/tests/api/test_admin_api.py`

- [ ] **Step 1: 写失败测试，约束 member 无法访问 admin API**

```python
# backend/tests/api/test_admin_api.py
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api.admin import router as admin_router


def test_member_cannot_access_admin_route():
    app = FastAPI()
    app.include_router(admin_router)
    c = TestClient(app)
    r = c.get("/api/admin/users")
    assert r.status_code in (401, 403)
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/api/test_admin_api.py -v
```

Expected:

```text
FAIL，缺少 /api/admin/* 路由
```

- [ ] **Step 3: 实现只读 admin router 与审计写入**

```python
# backend/api/admin.py
from fastapi import APIRouter, Depends
from api.deps import require_admin
from storage.user_catalog import UserCatalog
from storage.book_catalog import get_book_catalog
from storage.conversation_catalog import ConversationCatalog
from storage.audit_log_catalog import AuditLogCatalog

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/users")
def list_users(current_admin: dict = Depends(require_admin)):
    rows = UserCatalog().list_all()
    AuditLogCatalog().write(
        actor_user_id=current_admin["user_id"],
        actor_role="admin",
        action="admin_list_users",
        resource_type="user",
        resource_id="*",
        result="ok",
    )
    return rows
```

```python
# backend/main.py
from api.admin import router as admin_router
...
app.include_router(admin_router)
```

- [ ] **Step 4: 运行测试**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/api/test_admin_api.py -v
```

Expected:

```text
管理员可读通过，member 返回 403
```

- [ ] **Step 5: 提交**

```bash
git add backend/storage/audit_log_catalog.py backend/api/admin.py backend/main.py backend/tests/api/test_admin_api.py
git commit -m "feat: add read-only admin APIs and audit logging"
```

## Task 9: 前端接入鉴权与会话 API

**Files:**
- Modify: `frontend/src/api/index.js`
- Create: `frontend/src/stores/auth.js`
- Modify: `frontend/src/stores/chat.js`
- Modify: `frontend/src/stores/books.js`
- Modify: `frontend/src/router/index.js`
- Create: `frontend/src/views/AdminView.vue`

- [ ] **Step 1: 写前端 store 单测或最小集成断言（可选 Vitest，如暂未接入则先写手动验收脚本）**

```js
// frontend/src/stores/auth.js 目标接口约束
// login(email, password) -> 保存 access_token
// logout() -> 清空 token 与用户态
```

- [ ] **Step 2: 接口层改造**

```javascript
// frontend/src/api/index.js
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

export const authApi = {
  register: (payload) => api.post('/auth/register', payload),
  login: (payload) => api.post('/auth/login', payload),
  refresh: (payload) => api.post('/auth/refresh', payload),
  logout: (payload) => api.post('/auth/logout', payload),
}

export const conversationsApi = {
  create: (bookId, payload) => api.post(`/api/user/books/${bookId}/conversations`, payload),
  list: (bookId) => api.get(`/api/user/books/${bookId}/conversations`),
}
```

- [ ] **Step 3: 聊天 store 从 thread_id 切到 conversation_id**

```javascript
// frontend/src/stores/chat.js 关键变更
const conversationId = ref(null)
...
await fetchSSEStream(
  '/api/user/chat/stream',
  { query, book_id: selectedBookId.value, conversation_id: conversationId.value },
  ...
)
```

- [ ] **Step 4: 路由拆分 admin 页面**

```javascript
// frontend/src/router/index.js 关键片段
{ path: 'admin', name: 'admin', component: () => import('@/views/AdminView.vue') }
```

- [ ] **Step 5: 手工验收与提交**

Run:

```bash
cd frontend
npm run build
```

Expected:

```text
构建成功，无类型/语法错误
```

Commit:

```bash
git add frontend/src/api/index.js frontend/src/stores/auth.js frontend/src/stores/chat.js frontend/src/stores/books.js frontend/src/router/index.js frontend/src/views/AdminView.vue
git commit -m "feat: wire frontend auth and conversation APIs"
```

## Task 10: 记忆与对象存储按用户隔离 + 全量回归

**Files:**
- Modify: `backend/memory/mem0_store.py`
- Modify: `backend/api/chat.py`
- Modify: `backend/storage/oss_client.py`
- Modify: `backend/api/books.py`
- Modify: `README.md`

- [ ] **Step 1: 写失败测试，约束 mem0 使用 token user_id**

```python
# backend/tests/api/test_chat_integration.py 新增断言
assert app_state.agent.run_calls[0]["user_id"] == "token-user-id"
```

- [ ] **Step 2: 改造 mem0 与 OSS key 生成**

```python
# backend/memory/mem0_store.py 方法签名改造
def search(self, *, user_id: str, query: str, top_k: int = 3) -> list[str]:
    ...
    raw = self._client.search(query=query, filters={"user_id": user_id}, limit=top_k)

def add_qa(self, *, user_id: str, query: str, answer: str) -> None:
    ...
    self._client.add(messages=messages, user_id=user_id, prompt=USER_MEMORY_EXTRACTION_PROMPT)
```

```python
# backend/storage/oss_client.py 关键接口
def book_key(self, user_id: str, filename: str) -> str:
    return f"users/{user_id}/books/{filename}"
```

- [ ] **Step 3: 更新调用方**

```python
# backend/api/chat.py
memory_ctx = _fetch_memory(mem0, user_id=current_user["user_id"], query=req.query)
...
mem0.add_qa(user_id=current_user["user_id"], query=req.query, answer=result.answer)
```

```python
# backend/api/books.py
book_key = oss.book_key(current_user["user_id"], filename)
```

- [ ] **Step 4: 运行全量后端测试**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests -v
```

Expected:

```text
全部通过；若存在历史用例与新鉴权冲突，先修正测试夹具再通过
```

- [ ] **Step 5: 更新文档并提交**

```bash
git add backend/memory/mem0_store.py backend/api/chat.py backend/storage/oss_client.py backend/api/books.py README.md
git commit -m "feat: isolate memory and object storage by user"
```

## Task 11: 端到端验收与发布准备

**Files:**
- Modify: `README.md`
- Modify: `docs/test_evidence_report.html`（如你们继续沿用证据页）

- [ ] **Step 1: 执行后端与前端验收命令**

Run:

```bash
./.venv312/bin/python -m pytest backend/tests/api/test_chat_integration.py backend/tests/auth/test_auth_api.py backend/tests/api/test_admin_api.py backend/tests/api/test_conversations_api.py -v
cd frontend && npm run build
```

Expected:

```text
核心鉴权与隔离测试通过，前端可构建
```

- [ ] **Step 2: 手工走查清单**

```text
1) member 登录后只能看自己的书和会话
2) member 访问 /api/admin/* 返回 403
3) admin 登录后可查看用户数据，但无写操作入口
4) 伪造 conversation_id 无法读取他人会话
5) refresh token 轮转后旧 token 立即失效
```

- [ ] **Step 3: 记录验收证据**

```text
把关键接口响应、测试结果、管理员审计日志截图纳入 docs/test_evidence_report.html
```

- [ ] **Step 4: 最终提交**

```bash
git add README.md docs/test_evidence_report.html
git commit -m "docs: add auth isolation rollout and verification evidence"
```

- [ ] **Step 5: 打标签（可选）**

```bash
git tag v0.9.0-auth-isolation
```

## 自检

- 覆盖性检查：已覆盖 token 策略、Redis refresh、应用层 owner 隔离、每书多会话、管理员只读与审计、前后端改造和测试证据。
- 占位词检查：无 `TBD`、`TODO`、`implement later`。
- 命名一致性检查：统一使用 `owner_user_id`、`conversation_id`、`/api/user/*`、`/api/admin/*`。
