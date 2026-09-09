"""Repository for standing tool-approval rules ("trusted automation").

Each row is a durable grant — "auto-approve calls to `tool_name` whose
args match `args_pattern`" — plus when and why (`note`, `created_at`),
so the table itself is the audit trail of what's been pre-approved,
not just a runtime allowlist that forgets its own history.
`core.permissions.gate.TrustStoreGate` is what actually consults this
at approval time; `kanna trust ...` (`interfaces/cli/commands/
trust.py`) is how a user manages it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.memory.db import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrustRule:
    id: int
    tool_name: str
    args_pattern: dict = field(default_factory=dict)
    note: str = ""
    created_at: str = ""


class TrustRuleRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, *, tool_name: str, args_pattern: dict | None = None, note: str = "") -> TrustRule:
        pattern = args_pattern or {}
        created_at = _now()
        cur = self.db.execute(
            "INSERT INTO trust_rules (tool_name, args_pattern, note, created_at) VALUES (?, ?, ?, ?)",
            (tool_name, json.dumps(pattern), note, created_at),
        )
        return TrustRule(id=cur.lastrowid, tool_name=tool_name, args_pattern=pattern, note=note,
                          created_at=created_at)

    def list_all(self) -> list[TrustRule]:
        rows = self.db.query("SELECT * FROM trust_rules ORDER BY id ASC")
        return [_row_to_rule(r) for r in rows]

    def get(self, rule_id: int) -> TrustRule | None:
        row = self.db.query_one("SELECT * FROM trust_rules WHERE id = ?", (rule_id,))
        return None if row is None else _row_to_rule(row)

    def remove(self, rule_id: int) -> bool:
        """Returns True if a rule was actually deleted, False if `rule_id` didn't exist."""
        cur = self.db.execute("DELETE FROM trust_rules WHERE id = ?", (rule_id,))
        return cur.rowcount > 0


def _row_to_rule(row) -> TrustRule:
    return TrustRule(
        id=row["id"], tool_name=row["tool_name"],
        args_pattern=json.loads(row["args_pattern"] or "{}"),
        note=row["note"] or "", created_at=row["created_at"],
    )
