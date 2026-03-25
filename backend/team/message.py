"""MessageEnvelope and message type constants for the Agent Team.

Mirrors the JSONL message schema from examples/agents/s09-s10, but passed
as Python objects through threading.Queue instead of written to disk.
"""
from __future__ import annotations

import queue
from dataclasses import dataclass, field
from typing import Any

VALID_MSG_TYPES = {
    "task_request",
    "task_response",
    "task_error",
    "shutdown_request",
    "shutdown_response",
    "broadcast",
}


@dataclass
class MessageEnvelope:
    msg_type: str          # one of VALID_MSG_TYPES
    request_id: str        # full uuid4().hex — unique per envelope
    sender: str            # "dispatcher" | worker name
    payload: dict[str, Any]
    reply_to: "queue.Queue[MessageEnvelope] | None" = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.msg_type not in VALID_MSG_TYPES:
            raise ValueError(
                f"Invalid msg_type {self.msg_type!r}. Valid: {VALID_MSG_TYPES}"
            )
