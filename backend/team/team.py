"""AgentTeam — lifecycle manager for all Worker pools.

Mirrors s09 TeammateManager + config.json observability pattern,
and s10 shutdown_request/response FSM for graceful shutdown.
"""
from __future__ import annotations

import json
import queue
import sys
import time
import uuid
from pathlib import Path

from backend.agents.deepread_agent import DeepReadAgent
from backend.agents.note_agent import NoteAgent
from backend.agents.plan_editor import PlanEditor
from backend.agents.recommendation_agent import RecommendationAgent
from backend.config import get_settings
from backend.storage.note_vector_store import make_note_vector_store
from backend.team.message import MessageEnvelope
from backend.team.worker import WorkerPool
from backend.team.workers.deepread_worker import DeepReadWorker
from backend.team.workers.notes_worker import NotesWorker
from backend.team.workers.plan_worker import PlanWorker
from backend.team.workers.recommend_worker import RecommendWorker

_CONFIG_PATH = Path("data/team/config.json")

# Module-level singletons set by AgentTeam.startup()
_team: "AgentTeam | None" = None
_plan_editor: PlanEditor | None = None


def get_agent_team() -> "AgentTeam":
    if _team is None:
        raise RuntimeError("AgentTeam has not been started — call startup() first")
    return _team


def get_plan_editor() -> PlanEditor:
    if _plan_editor is None:
        raise RuntimeError("AgentTeam has not been started — call startup() first")
    return _plan_editor


class AgentTeam:
    """Manages persistent Worker pools and their lifecycle.

    Usage (FastAPI lifespan):
        team = AgentTeam()
        team.startup()
        yield
        team.shutdown()
    """

    def __init__(self) -> None:
        self.pools: dict[str, WorkerPool] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start all Worker threads. Called once at app startup."""
        global _team, _plan_editor

        settings = get_settings()
        note_vector_store = make_note_vector_store(settings)

        # Shared PlanEditor singleton (also exposed via get_plan_editor() for reader.py)
        plan_editor = PlanEditor()
        _plan_editor = plan_editor

        # One shared NoteAgent instance for DeepRead auto-hooks
        shared_note_agent = NoteAgent(note_vector_store=note_vector_store)

        self.pools = {
            "deepread": WorkerPool(
                "deepread",
                lambda: DeepReadWorker(DeepReadAgent(), shared_note_agent),
                size=2,
            ),
            "notes": WorkerPool(
                "notes",
                lambda: NotesWorker(NoteAgent()),
                size=2,
            ),
            "plan": WorkerPool(
                "plan",
                lambda: PlanWorker(plan_editor),
                size=1,
            ),
            "recommend": WorkerPool(
                "recommend",
                lambda: RecommendWorker(RecommendationAgent()),
                size=2,
            ),
        }

        for pool in self.pools.values():
            for worker in pool.all_workers():
                worker.start()

        _team = self
        self._write_config("running")
        print("[AgentTeam] all workers started", file=sys.stdout)

    def shutdown(self) -> None:
        """Graceful shutdown via s10 shutdown_request/response FSM."""
        reply_queue: queue.Queue[MessageEnvelope] = queue.Queue()
        all_workers = [w for pool in self.pools.values() for w in pool.all_workers()]

        # Broadcast shutdown_request to every worker
        req_map: dict[str, str] = {}
        for worker in all_workers:
            req_id = uuid.uuid4().hex
            req_map[req_id] = worker.name
            worker.inbox.put(MessageEnvelope(
                msg_type="shutdown_request",
                request_id=req_id,
                sender="team",
                payload={},
                reply_to=reply_queue,
            ))

        # Wait up to 5 s for all acknowledgements
        deadline = time.monotonic() + 5.0
        acked = 0
        while acked < len(all_workers):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                reply_queue.get(timeout=remaining)
                acked += 1
            except queue.Empty:
                break

        if acked < len(all_workers):
            print(
                f"[AgentTeam] {len(all_workers) - acked} worker(s) did not ack shutdown",
                file=sys.stderr,
            )

        for worker in all_workers:
            worker.join(timeout=1.0)

        self._write_config("shutdown")
        print("[AgentTeam] shutdown complete", file=sys.stdout)

    # ------------------------------------------------------------------
    # BM25 cache invalidation
    # ------------------------------------------------------------------

    def invalidate_bm25_caches(self) -> None:
        """Called by POST /books/upload after successful ingest."""
        for pool in self.pools.values():
            for worker in pool.all_workers():
                worker.invalidate_bm25()

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def _write_config(self, status: str) -> None:
        """Write team status to data/team/config.json (s09 config.json pattern)."""
        try:
            _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            members = []
            for pool in self.pools.values():
                for w in pool.all_workers():
                    members.append({"name": w.name, "role": w.role, "status": w.status})
            _CONFIG_PATH.write_text(
                json.dumps({"status": status, "members": members}, indent=2)
            )
        except Exception as exc:
            print(f"[AgentTeam] config write failed: {exc}", file=sys.stderr)
