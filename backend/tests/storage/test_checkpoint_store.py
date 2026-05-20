from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from storage.checkpoint_store import CheckpointStore, build_thread_id


def test_build_thread_id_uses_conversation_id():
    assert build_thread_id(conversation_id="conv-1") == "conv-1"


def test_create_sync_checkpointer_requires_postgres_dsn():
    store = CheckpointStore(settings=SimpleNamespace(postgres_dsn=""))

    try:
        with store.create_sync_checkpointer():
            pass
    except RuntimeError as exc:
        assert "POSTGRES_DSN or DATABASE_URL is required" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when POSTGRES_DSN is missing.")


def test_create_sync_checkpointer_uses_postgres_when_dsn_present(monkeypatch):
    created = []

    @contextmanager
    def fake_postgres():
        created.append("postgres")
        yield object()

    monkeypatch.setattr(CheckpointStore, "_create_postgres_checkpointer", lambda self: fake_postgres())

    store = CheckpointStore(settings=SimpleNamespace(postgres_dsn="postgresql://example"))
    with store.create_sync_checkpointer():
        pass

    assert created == ["postgres"]


def test_create_sync_checkpointer_uses_database_url(monkeypatch):
    created = []

    @contextmanager
    def fake_postgres():
        created.append("postgres")
        yield object()

    monkeypatch.setattr(CheckpointStore, "_create_postgres_checkpointer", lambda self: fake_postgres())

    store = CheckpointStore(settings=SimpleNamespace(postgres_dsn="", database_url="postgresql://example"))
    with store.create_sync_checkpointer():
        pass

    assert created == ["postgres"]


def test_create_async_checkpointer_uses_postgres_when_dsn_present(monkeypatch):
    created = []

    @asynccontextmanager
    async def fake_postgres():
        created.append("postgres-async")
        yield object()

    monkeypatch.setattr(CheckpointStore, "_create_async_postgres_checkpointer", lambda self: fake_postgres())

    store = CheckpointStore(settings=SimpleNamespace(postgres_dsn="postgresql://example"))

    async def _run():
        async with store.create_async_checkpointer():
            pass

    import asyncio

    asyncio.run(_run())
    assert created == ["postgres-async"]


def test_get_chat_history_normalizes_messages(monkeypatch):
    saver = SimpleNamespace(
        setup=lambda: None,
        get_tuple=lambda config: SimpleNamespace(
            checkpoint={
                "channel_values": {
                    "messages": [
                        HumanMessage(content="【用户问题】：\nWhat is duty?"),
                        AIMessage(content="Duty is practical necessity."),
                    ]
                }
            }
        ),
    )

    @contextmanager
    def fake_checkpointer():
        yield saver

    monkeypatch.setattr(CheckpointStore, "create_sync_checkpointer", lambda self: fake_checkpointer())

    store = CheckpointStore(settings=SimpleNamespace(postgres_dsn=""))
    history = store.get_chat_history(conversation_id="conv-1")

    assert history == [
        {"role": "user", "content": "What is duty?"},
        {"role": "ai", "content": "Duty is practical necessity."},
    ]


def test_clear_chat_history_deletes_thread(monkeypatch):
    deleted = []
    saver = SimpleNamespace(
        setup=lambda: None,
        delete_thread=lambda thread_id: deleted.append(thread_id),
    )

    @contextmanager
    def fake_checkpointer():
        yield saver

    monkeypatch.setattr(CheckpointStore, "create_sync_checkpointer", lambda self: fake_checkpointer())

    store = CheckpointStore(settings=SimpleNamespace(postgres_dsn=""))
    store.clear_chat_history(conversation_id="conv-1")

    assert deleted == ["conv-1"]


def test_add_ai_message_updates_state_with_agent_builder(monkeypatch):
    update_calls = []

    class FakeReactAgent:
        def update_state(self, config, values):
            update_calls.append((config, values))

    saver = SimpleNamespace(setup=lambda: None)

    @contextmanager
    def fake_checkpointer():
        yield saver

    monkeypatch.setattr(CheckpointStore, "create_sync_checkpointer", lambda self: fake_checkpointer())

    store = CheckpointStore(settings=SimpleNamespace(postgres_dsn=""))
    store.add_ai_message(
        "critic message",
        conversation_id="conv-1",
        agent_builder=lambda *, memory, locale=None: (FakeReactAgent(), []),
    )

    assert len(update_calls) == 1
    config, values = update_calls[0]
    assert config == {"configurable": {"thread_id": "conv-1"}}
    assert values["messages"][0].content == "critic message"


def test_add_ai_message_without_agent_builder_is_tolerated(monkeypatch):
    saver = SimpleNamespace(setup=lambda: None)

    @contextmanager
    def fake_checkpointer():
        yield saver

    monkeypatch.setattr(CheckpointStore, "create_sync_checkpointer", lambda self: fake_checkpointer())

    store = CheckpointStore(settings=SimpleNamespace(postgres_dsn=""))
    store.add_ai_message("critic message", conversation_id="conv-1")
