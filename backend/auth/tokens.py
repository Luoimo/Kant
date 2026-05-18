from __future__ import annotations

from datetime import datetime, timedelta, timezone
import uuid

import jwt

from config import get_settings


def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def create_access_token(*, user_id: str, role: str) -> str:
    settings = get_settings()
    now = _now_ts()
    exp = int((datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_token_minutes)).timestamp())
    payload = {
        "sub": user_id,
        "role": role,
        "jti": str(uuid.uuid4()),
        "iss": settings.jwt_issuer,
        "iat": now,
        "exp": exp,
        "tenant_id": "default",
        "typ": "access",
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")


def create_refresh_token(*, user_id: str, role: str) -> tuple[str, str]:
    settings = get_settings()
    jti = str(uuid.uuid4())
    now = _now_ts()
    exp = int((datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_token_days)).timestamp())
    payload = {
        "sub": user_id,
        "role": role,
        "jti": jti,
        "iss": settings.jwt_issuer,
        "iat": now,
        "exp": exp,
        "tenant_id": "default",
        "typ": "refresh",
    }
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")
    return token, jti


def decode_token(token: str) -> dict:
    settings = get_settings()
    return jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=["HS256"],
        issuer=settings.jwt_issuer,
    )
