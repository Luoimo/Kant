from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr

from auth.passwords import hash_password, verify_password
from auth.refresh_store import RefreshStore
from auth.tokens import create_access_token, create_refresh_token, decode_token
from storage.audit_log_catalog import AuditLogCatalog
from storage.user_catalog import UserCatalog


router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


@router.post("/register")
def register(req: RegisterRequest) -> dict:
    if len(req.password) < 8:
        raise HTTPException(status_code=422, detail="password too short")
    catalog = UserCatalog()
    existing = catalog.get_by_email(req.email)
    if existing:
        raise HTTPException(status_code=409, detail="email already exists")

    user = catalog.create_member(email=req.email, password_hash=hash_password(req.password))
    AuditLogCatalog().write(
        actor_user_id=user["user_id"],
        actor_role=user["role"],
        action="register",
        resource_type="user",
        resource_id=user["user_id"],
        result="ok",
    )
    return {"user_id": user["user_id"], "email": user["email"], "role": user["role"]}


@router.post("/login")
def login(req: LoginRequest) -> dict:
    user = UserCatalog().get_by_email(req.email)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="invalid credentials")
    access_token = create_access_token(user_id=user["user_id"], role=user["role"])
    refresh_token, refresh_jti = create_refresh_token(user_id=user["user_id"], role=user["role"])
    RefreshStore().save(user_id=user["user_id"], jti=refresh_jti, token=refresh_token)
    AuditLogCatalog().write(
        actor_user_id=user["user_id"],
        actor_role=user["role"],
        action="login",
        resource_type="session",
        resource_id=refresh_jti,
        result="ok",
    )
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "role": user["role"],
        "user_id": user["user_id"],
    }


@router.post("/refresh")
def refresh(req: RefreshRequest) -> dict:
    payload = decode_token(req.refresh_token)
    if payload.get("typ") != "refresh":
        raise HTTPException(status_code=401, detail="invalid token type")

    user_id = payload["sub"]
    role = payload["role"]
    jti = payload["jti"]
    store = RefreshStore()
    if not store.verify(user_id=user_id, jti=jti, token=req.refresh_token):
        raise HTTPException(status_code=401, detail="refresh token revoked")

    store.revoke(user_id=user_id, jti=jti)
    new_access = create_access_token(user_id=user_id, role=role)
    new_refresh, new_jti = create_refresh_token(user_id=user_id, role=role)
    store.save(user_id=user_id, jti=new_jti, token=new_refresh)
    return {
        "access_token": new_access,
        "refresh_token": new_refresh,
        "token_type": "bearer",
    }


@router.post("/logout")
def logout(req: LogoutRequest) -> dict:
    payload = decode_token(req.refresh_token)
    if payload.get("typ") != "refresh":
        raise HTTPException(status_code=401, detail="invalid token type")
    RefreshStore().revoke(user_id=payload["sub"], jti=payload["jti"])
    return {"status": "ok"}


@router.post("/logout-all")
def logout_all(req: RefreshRequest) -> dict:
    payload = decode_token(req.refresh_token)
    if payload.get("typ") != "refresh":
        raise HTTPException(status_code=401, detail="invalid token type")
    RefreshStore().revoke_all(user_id=payload["sub"])
    return {"status": "ok"}
