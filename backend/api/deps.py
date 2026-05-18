from __future__ import annotations

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth.tokens import decode_token


bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict:
    if creds is None:
        raise HTTPException(status_code=401, detail="missing token")
    try:
        payload = decode_token(creds.credentials)
    except Exception as exc:  # pragma: no cover - jwt error branch
        raise HTTPException(status_code=401, detail=f"invalid token: {exc}") from exc

    if payload.get("typ") != "access":
        raise HTTPException(status_code=401, detail="invalid token type")

    return {"user_id": payload["sub"], "role": payload["role"]}


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="admin required")
    return user


def require_member(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "member":
        raise HTTPException(status_code=403, detail="member required")
    return user
