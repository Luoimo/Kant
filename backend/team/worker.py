"""AgentWorker base class and WorkerPool.

Inspired by s09 _teammate_loop and s10 shutdown FSM — each Worker is a
persistent daemon thread with its own inbox Queue and per-thread_id message
history. WorkerPool holds N workers of the same type.
"""
from __future__ import annotations

import queue
import threading
from typing import Any, Callable

from backend.team.message import MessageEnvelope

_HISTORY_MAX = 20   # max messages kept per thread_id


class AgentWorker(threading.Thread):
    """Persistent daemon thread wrapping an agent.

    Subclasses implement _execute(payload) -> dict.
    Communication happens via self.inbox (MessageEnvelope objects).
    """

    def __init__(self, name: str, role: str) -> None:
        super().__init__(name=name, daemon=True)
        self.role = role
        self.inbox: queue.Queue[MessageEnvelope] = queue.Queue()
        self.status: str = "idle"
        # Per-session message history keyed by thread_id (s09 _teammate_loop pattern)
        self._histories: dict[str, list[dict[str, Any]]] = {}
        self._shutdown_event = threading.Event()

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                msg = self.inbox.get(timeout=1.0)
            except queue.Empty:
                continue
            self._handle(msg)

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    def _handle(self, msg: MessageEnvelope) -> None:
        if msg.msg_type == "task_request":
            self.status = "working"
            try:
                result = self._execute(msg.payload)
                response = MessageEnvelope(
                    msg_type="task_response",
                    request_id=msg.request_id,
                    sender=self.name,
                    payload=result,
                )
            except Exception as exc:
                response = MessageEnvelope(
                    msg_type="task_error",
                    request_id=msg.request_id,
                    sender=self.name,
                    payload={"error": str(exc)},
                )
            finally:
                self.status = "idle"
            if msg.reply_to is not None:
                msg.reply_to.put(response)

        elif msg.msg_type == "shutdown_request":
            # s10 FSM: respond before setting event so reply_to.put() succeeds
            if msg.reply_to is not None:
                msg.reply_to.put(MessageEnvelope(
                    msg_type="shutdown_response",
                    request_id=msg.request_id,
                    sender=self.name,
                    payload={"approve": True},
                ))
            self._shutdown_event.set()
            self.status = "shutdown"

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def _execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def invalidate_bm25(self) -> None:
        """Override in subclasses that hold a retriever with a BM25 cache."""
        pass  # no-op by default; not a silent failure

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _get_history(self, thread_id: str) -> list[dict[str, Any]]:
        return self._histories.setdefault(thread_id, [])

    def _trim_history(self, thread_id: str) -> None:
        h = self._histories.get(thread_id)
        if h and len(h) > _HISTORY_MAX:
            self._histories[thread_id] = h[-_HISTORY_MAX:]


class WorkerPool:
    """Holds N AgentWorker threads of the same type.

    worker_factory is a zero-argument callable returning a fully constructed
    AgentWorker (not a bare agent), so multi-arg constructors are handled at
    the call site via lambda or functools.partial.
    """

    def __init__(
        self,
        name: str,
        worker_factory: Callable[[], AgentWorker],
        size: int = 2,
    ) -> None:
        self.name = name
        self.workers: list[AgentWorker] = [worker_factory() for _ in range(size)]

    def least_busy(self) -> AgentWorker:
        """Return first idle worker, or the one with the shortest inbox queue."""
        idle = [w for w in self.workers if w.status == "idle"]
        if idle:
            return idle[0]
        return min(self.workers, key=lambda w: w.inbox.qsize())

    def all_workers(self) -> list[AgentWorker]:
        return self.workers
