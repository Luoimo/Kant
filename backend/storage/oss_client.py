"""阿里云 OSS 客户端封装。

提供 EPUB 原文件与封面图片的云端持久化，并为 book_catalog 中的
`source`/`cover_path` 字段约定统一的 ``oss://{bucket}/{key}`` 表示。

设计目标：
1. 把本地文件系统操作 (``Path.write_bytes`` / ``unlink``) 抽象成同样的接口，
   调用方只需关心 ``key``，不直接依赖 oss2。
2. 当 OSS 配置缺失时（本地开发常见），自动 ``enabled = False``，调用方可据此
   降级到本地文件系统。
3. 对外暴露 ``signed_url``，给前端直接读取私有 bucket 中的对象。
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

try:
    import oss2  # type: ignore
except ImportError:  # pragma: no cover - 运行环境必须安装 oss2
    oss2 = None  # type: ignore

from config import get_settings

logger = logging.getLogger(__name__)


OSS_URI_SCHEME = "oss://"


def is_oss_uri(value: str | None) -> bool:
    """判断字符串是否是 ``oss://bucket/key`` 形式。"""
    return bool(value) and value.startswith(OSS_URI_SCHEME)  # type: ignore[arg-type]


def parse_oss_uri(uri: str) -> tuple[str, str]:
    """解析 ``oss://bucket/key`` -> ``(bucket, key)``。"""
    if not is_oss_uri(uri):
        raise ValueError(f"非法 OSS URI: {uri}")
    without_scheme = uri[len(OSS_URI_SCHEME):]
    bucket, _, key = without_scheme.partition("/")
    if not bucket or not key:
        raise ValueError(f"OSS URI 缺少 bucket 或 key: {uri}")
    return bucket, key


def build_oss_uri(bucket: str, key: str) -> str:
    return f"{OSS_URI_SCHEME}{bucket}/{key}"


class OSSClient:
    """轻量封装 oss2.Bucket，集中管理 access key / endpoint / bucket。"""

    def __init__(self) -> None:
        settings = get_settings()
        self.access_key_id = settings.oss_access_key_id
        self.access_key_secret = settings.oss_secret_access_key
        self.endpoint = settings.oss_endpoint
        self.bucket_name = settings.oss_bucket
        self.books_prefix = settings.oss_books_prefix
        self.covers_prefix = settings.oss_covers_prefix
        self.signed_url_expires = settings.oss_signed_url_expires

        self._bucket: Optional["oss2.Bucket"] = None
        if not self.enabled:
            logger.warning(
                "OSS 未启用：access_key_id/secret/bucket 任一为空，调用方应降级到本地文件系统"
            )
            return

        if oss2 is None:
            raise RuntimeError("oss2 未安装，请在 requirements.txt 中启用 oss2>=2.19.0")

        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self._bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)

    # ------------------------------------------------------------------
    # 状态判断
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self.access_key_id and self.access_key_secret and self.bucket_name)

    # ------------------------------------------------------------------
    # Key 构造
    # ------------------------------------------------------------------

    def book_key(self, user_id: str, filename: str) -> str:
        return f"users/{user_id}/books/{filename}"

    def cover_key(self, user_id: str, book_id: str, ext: str) -> str:
        ext = ext if ext.startswith(".") else f".{ext}"
        return f"users/{user_id}/covers/{book_id}{ext}"

    # ------------------------------------------------------------------
    # 基本 IO
    # ------------------------------------------------------------------

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> str:
        """上传字节流，返回 ``oss://bucket/key``。"""
        assert self._bucket is not None, "OSS 未启用"
        headers = {"Content-Type": content_type} if content_type else None
        self._bucket.put_object(key, data, headers=headers)
        return build_oss_uri(self.bucket_name, key)

    def put_file(self, key: str, local_path: str | Path, content_type: str | None = None) -> str:
        """上传本地文件，返回 ``oss://bucket/key``。"""
        assert self._bucket is not None, "OSS 未启用"
        headers = {"Content-Type": content_type} if content_type else None
        self._bucket.put_object_from_file(key, str(local_path), headers=headers)
        return build_oss_uri(self.bucket_name, key)

    def get_to_file(self, key: str, local_path: str | Path) -> None:
        """下载对象到本地路径。"""
        assert self._bucket is not None, "OSS 未启用"
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._bucket.get_object_to_file(key, str(local_path))

    def delete(self, key: str) -> bool:
        """删除对象，返回是否真实删除（对象不存在返回 False）。"""
        assert self._bucket is not None, "OSS 未启用"
        try:
            if not self._bucket.object_exists(key):
                return False
            self._bucket.delete_object(key)
            return True
        except Exception:
            logger.exception("OSS 删除对象失败: %s", key)
            return False

    def exists(self, key: str) -> bool:
        assert self._bucket is not None, "OSS 未启用"
        return bool(self._bucket.object_exists(key))

    # ------------------------------------------------------------------
    # 签名 URL
    # ------------------------------------------------------------------

    def signed_url(self, key: str, expires: int | None = None) -> str:
        """生成带签名的临时 URL，前端可直接 GET 读取。"""
        assert self._bucket is not None, "OSS 未启用"
        ttl = expires if expires is not None else self.signed_url_expires
        # slash_safe=True：避免 key 中的 `/` 被转义，保持路径可读
        return self._bucket.sign_url("GET", key, ttl, slash_safe=True)

    def signed_url_from_uri(self, oss_uri: str, expires: int | None = None) -> str:
        """``oss://bucket/key`` 直接换成签名 URL。"""
        bucket, key = parse_oss_uri(oss_uri)
        if bucket != self.bucket_name:
            raise ValueError(
                f"OSS URI bucket ({bucket}) 与当前客户端 bucket ({self.bucket_name}) 不一致"
            )
        return self.signed_url(key, expires=expires)


@lru_cache(maxsize=1)
def get_oss_client() -> OSSClient:
    return OSSClient()


__all__ = [
    "OSSClient",
    "get_oss_client",
    "is_oss_uri",
    "parse_oss_uri",
    "build_oss_uri",
    "OSS_URI_SCHEME",
]
