"""Repository for conversation/session state."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.memory.db import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Session:
    id: str
    started_at: str
    ended_at: str | None
    metadata: dict[str, Any]


@dataclass
class Message:
    id: int
    session_id: str
    role: str
    content: str
    created_at: str


class SessionRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, metadata: dict[str, Any] | None = None) -> Session:
        session_id = str(uuid.uuid4())
        started_at = _now()
        self.db.execute(
            "INSERT INTO sessions (id, started_at, ended_at, metadata) VALUES (?, ?, NULL, ?)",
            (session_id, started_at, json.dumps(metadata or {})),
        )
        return Session(id=session_id, started_at=started_at, ended_at=None, metadata=metadata or {})

    def end(self, session_id: str) -> None:
        self.db.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (_now(), session_id))

    def get(self, session_id: str) -> Session | None:
        row = self.db.query_one("SELECT * FROM sessions WHERE id = ?", (session_id,))
        if row is None:
            return None
        return Session(id=row["id"], started_at=row["started_at"], ended_at=row["ended_at"],
                        metadata=json.loads(row["metadata"]))

    def add_message(self, session_id: str, role: str, content: str) -> Message:
        created_at = _now()
        cur = self.db.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, created_at),
        )
        return Message(id=cur.lastrowid, session_id=session_id, role=role, content=content,
                        created_at=created_at)

    def get_messages(self, session_id: str) -> list[Message]:
        rows = self.db.query(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,)
        )
        return [Message(id=r["id"], session_id=r["session_id"], role=r["role"], content=r["content"],
                         created_at=r["created_at"]) for r in rows]
