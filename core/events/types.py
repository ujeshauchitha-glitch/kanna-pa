"""Event types published on Kanna's internal event bus."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Event:
    """A single fact published on the bus.

    `topic` is a dotted string such as `tool.executed`, `agent.state_changed`,
    or `finance.transaction_created` — subscribers filter by prefix.
    """

    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)
    session_id: str | None = None
