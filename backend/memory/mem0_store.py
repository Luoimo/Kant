from __future__ import annotations
from dotenv import load_dotenv

import logging

logger = logging.getLogger(__name__)

load_dotenv()



class Mem0Store:
    """
    Mem0 长期记忆封装。

    - search / add_qa / delete_all 内部捕获所有异常并降级，不影响主流程。
    """

    def __init__(self) -> None:
        from backend.config import get_settings
        s = get_settings()
        # 配置项
        config = {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "mem0_vector_base",
                    # Optional: ChromaDB Cloud configuration
                    "api_key": s.chroma_api_key,
                    "tenant": s.chroma_tenant,
                }
            }
        }

        try:
            from mem0 import Memory
            memory = Memory.from_config(config)

            self._client = memory
            self._user_id = s.mem0_user_id.strip()
            self._enabled = True
            logger.info("Mem0 初始化成功（user_id=%s）", self._user_id)
        except ImportError:
            logger.warning("mem0ai 未安装，记忆功能已禁用（pip install mem0ai）")
            self._enabled = False
        except Exception as exc:
            logger.warning("Mem0 初始化失败（%s），记忆功能已禁用", exc)
            self._enabled = False

    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """检索与 query 相关的历史记忆，返回纯文本列表。"""
        if not self._enabled:
            return []
        try:
            raw = self._client.search(
                query=query,
                user_id=self._user_id
            )
            logger.debug("Mem0 search 原始返回类型=%s 内容=%s", type(raw).__name__, raw)
            # Cloud API 返回直接列表；少数版本可能包在 {"results":[]} 或 {"memories":[]} 里
            if isinstance(raw, list):
                items = raw
            elif isinstance(raw, dict):
                items = raw.get("results") or raw.get("memories") or []
            else:
                items = []
            return [r["memory"] for r in items if isinstance(r, dict) and "memory" in r]
        except Exception as exc:
            logger.warning("Mem0 search 失败：%s", exc)
            return []

    def add_qa(self, query: str, answer: str) -> None:
        """将一次完整问答存入长期记忆。"""
        if not self._enabled:
            return
        try:
            messages = [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer},
            ]
            result = self._client.add(
                messages=messages,
                user_id=self._user_id,
            )
            logger.debug("Mem0 add_qa 返回：%s", result)
        except Exception as exc:
            logger.warning("Mem0 add_qa 失败：%s", exc)

    def delete_all(self) -> None:
        """清空当前用户的所有记忆。"""
        if not self._enabled:
            return
        try:
            self._client.delete_all(user_id=self._user_id)
            logger.info("Mem0 已清空用户 %s 的所有记忆", self._user_id)
        except Exception as exc:
            logger.warning("Mem0 delete_all 失败：%s", exc)
