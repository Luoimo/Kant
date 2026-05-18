from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.auth import router as auth_router
from auth.passwords import hash_password, verify_password
from auth.tokens import create_access_token, decode_token
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


def test_password_hash_and_verify():
    h = hash_password("secret-123")
    assert h != "secret-123"
    assert verify_password("secret-123", h) is True
    assert verify_password("wrong", h) is False


def test_access_token_roundtrip():
    token = create_access_token(user_id="u-1", role="member")
    payload = decode_token(token)
    assert payload["sub"] == "u-1"
    assert payload["role"] == "member"


def test_auth_router_mounts():
    # 仅验证路由对象可导入与挂载，不触发真实数据库连接
    app = FastAPI()
    app.include_router(auth_router)
    paths = {route.path for route in app.routes}
    assert "/auth/login" in paths
