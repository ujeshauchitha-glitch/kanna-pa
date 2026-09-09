"""Repository for the tool execution audit log.

Every tool invocation the registry performs is recorded here — this is
what makes agent runs inspectable after the fact.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from core.memory.db import Database
from core.tools.result import ToolResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExecutionLogRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record(self, *, session_id: str | None, tool_name: str, args: dict, result: ToolResult) -> int:
        cur = self.db.execute(
            "INSERT INTO execution_log (session_id, tool_name, args, result, success, duration_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, tool_name, json.dumps(args), json.dumps(result.to_dict()),
             1 if result.success else 0, result.duration_ms, _now()),
        )
        return cur.lastrowid

    def recent(self, limit: int = 50) -> list[dict]:
        rows = self.db.query("SELECT * FROM execution_log ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def for_session(self, session_id: str) -> list[dict]:
        rows = self.db.query(
            "SELECT * FROM execution_log WHERE session_id = ? ORDER BY id ASC", (session_id,)
        )
        return [dict(r) for r in rows]
