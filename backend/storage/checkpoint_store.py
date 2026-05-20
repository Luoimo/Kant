from __future__ import annotations

import logging
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from config import Settings, get_settings


logger = logging.getLogger(__name__)


def build_thread_id(*, conversation_id: str) -> str:
    return conversation_id


def _extract_messages(checkpoint_tuple: Any) -> list[BaseMessage]:
    if not checkpoint_tuple:
        return []
    checkpoint = getattr(checkpoint_tuple, "checkpoint", None) or {}
    state = checkpoint.get("channel_values", {})
    messages = state.get("messages", [])
    return [message for message in messages if isinstance(message, BaseMessage)]


def _normalize_history_message(message: BaseMessage) -> dict[str, str] | None:
    import re

    if message.type in ["human", "user"]:
        content = message.content
        if "【用户问题】：\n" in content:
            content = content.split("【用户问题】：\n")[-1]
        elif "[历史阅读记录（仅供参考）]" in content:
            content = content.split("\n\n[历史阅读记录（仅供参考）]")[0]
            content = re.sub(r"^\[当前书籍来源：.*?\]\n\n", "", content)
        else:
            content = re.sub(r"^\[当前书籍来源：.*?\]\n\n", "", content)
            content = re.sub(r"^【当前阅读书籍】：.*?\n\n", "", content)
            content = re.sub(r"\n\n【当前阅读章节】：.*$", "", content)
        return {"role": "user", "content": str(content).strip()}

    if message.type in ["ai", "assistant"]:
        if not message.content and getattr(message, "tool_calls", None):
            return None
        return {"role": "ai", "content": str(message.content)}

    return None


class CheckpointStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def create_sync_checkpointer(self) -> AbstractContextManager[Any]:
        self._require_postgres_dsn()
        return self._create_postgres_checkpointer()

    def create_async_checkpointer(self) -> AbstractAsyncContextManager[Any]:
        self._require_postgres_dsn()
        return self._create_async_postgres_checkpointer()

    def get_chat_history(self, *, conversation_id: str) -> list[dict[str, str]]:
        actual_thread = build_thread_id(conversation_id=conversation_id)

        try:
            with self.create_sync_checkpointer() as memory:
                setup = getattr(memory, "setup", None)
                if callable(setup):
                    setup()

                tuple_ = memory.get_tuple({"configurable": {"thread_id": actual_thread}})
                history: list[dict[str, str]] = []
                for message in _extract_messages(tuple_):
                    item = _normalize_history_message(message)
                    if item:
                        history.append(item)
                return history
        except Exception as exc:
            logger.error("Failed to load chat history: %s", exc)
            return []

    def clear_chat_history(self, *, conversation_id: str) -> None:
        actual_thread = build_thread_id(conversation_id=conversation_id)

        try:
            with self.create_sync_checkpointer() as memory:
                setup = getattr(memory, "setup", None)
                if callable(setup):
                    setup()
                memory.delete_thread(actual_thread)
                logger.info("Successfully cleared chat history for thread %s", actual_thread)
        except Exception as exc:
            logger.error("Failed to clear chat history: %s", exc)

    def add_ai_message(
        self,
        content: str,
        *,
        conversation_id: str,
        agent_builder=None,
        locale: str | None = None,
    ) -> None:
        actual_thread = build_thread_id(conversation_id=conversation_id)

        try:
            with self.create_sync_checkpointer() as memory:
                setup = getattr(memory, "setup", None)
                if callable(setup):
                    setup()

                if agent_builder is None:
                    raise ValueError("agent_builder is required to append AI messages.")

                react_agent, _ = agent_builder(memory=memory, locale=locale)
                react_agent.update_state(
                    {"configurable": {"thread_id": actual_thread}},
                    {"messages": [AIMessage(content=content)]},
                )
                logger.info("Successfully appended AI message to thread %s", actual_thread)
        except Exception as exc:
            logger.error("Failed to append AI message to history: %s", exc)

    def _require_postgres_dsn(self) -> None:
        if not self._postgres_dsn():
            raise RuntimeError("POSTGRES_DSN or DATABASE_URL is required for checkpoint storage.")

    def _postgres_dsn(self) -> str:
        return self.settings.postgres_dsn or getattr(self.settings, "database_url", "")

    def _create_postgres_checkpointer(self) -> AbstractContextManager[Any]:
        from langgraph.checkpoint.postgres import PostgresSaver

        return PostgresSaver.from_conn_string(self._postgres_dsn())

    def _create_async_postgres_checkpointer(self) -> AbstractAsyncContextManager[Any]:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        return AsyncPostgresSaver.from_conn_string(self._postgres_dsn())


def get_checkpoint_store(settings: Settings | None = None) -> CheckpointStore:
    return CheckpointStore(settings=settings)


__all__ = ["CheckpointStore", "build_thread_id", "get_checkpoint_store"]
