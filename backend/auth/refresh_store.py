from __future__ import annotations

import hashlib
from datetime import timedelta

from redis import Redis

from config import get_settings


class RefreshStore:
    def __init__(self, client: Redis | None = None) -> None:
        settings = get_settings()
        self._redis = client or Redis.from_url(settings.redis_url, decode_responses=True)
        self._ttl_seconds = int(timedelta(days=settings.jwt_refresh_token_days).total_seconds())

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def save(self, *, user_id: str, jti: str, token: str) -> None:
        key = f"auth:refresh:{user_id}:{jti}"
        self._redis.hset(key, mapping={"token_hash": self._hash(token)})
        self._redis.expire(key, self._ttl_seconds)
        self._redis.sadd(f"auth:user_sessions:{user_id}", jti)

    def verify(self, *, user_id: str, jti: str, token: str) -> bool:
        key = f"auth:refresh:{user_id}:{jti}"
        data = self._redis.hgetall(key)
        if not data:
            return False
        return data.get("token_hash", "") == self._hash(token)

    def revoke(self, *, user_id: str, jti: str) -> None:
        key = f"auth:refresh:{user_id}:{jti}"
        self._redis.delete(key)
        self._redis.srem(f"auth:user_sessions:{user_id}", jti)

    def revoke_all(self, *, user_id: str) -> None:
        set_key = f"auth:user_sessions:{user_id}"
        jtis = self._redis.smembers(set_key)
        for jti in jtis:
            self._redis.delete(f"auth:refresh:{user_id}:{jti}")
        self._redis.delete(set_key)
