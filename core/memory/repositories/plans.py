"""Repository for persisted plans and their steps.

The agent loop writes here so a run's plan and per-step outcomes are
inspectable after the fact, independent of the `execution_log` (which
records every tool call regardless of whether it belongs to a plan).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from core.memory.db import Database
from core.planner.plan import Plan


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PlanRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, plan: Plan, *, session_id: str | None) -> str:
        plan_id = str(uuid.uuid4())
        now = _now()
        self.db.execute(
            "INSERT INTO plans (id, session_id, request, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'in_progress', ?, ?)",
            (plan_id, session_id, plan.request, now, now),
        )
        for i, step in enumerate(plan.steps):
            self.db.execute(
                "INSERT INTO plan_steps (id, plan_id, step_index, tool_name, args, expected, "
                " status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                (str(uuid.uuid4()), plan_id, i, step.tool_name, json.dumps(step.args),
                 json.dumps(step.expected), now, now),
            )
        return plan_id

    def record_step_result(self, plan_id: str, step_index: int, *, status: str, result: dict) -> None:
        self.db.execute(
            "UPDATE plan_steps SET status = ?, result = ?, updated_at = ? "
            "WHERE plan_id = ? AND step_index = ?",
            (status, json.dumps(result), _now(), plan_id, step_index),
        )

    def set_status(self, plan_id: str, status: str) -> None:
        self.db.execute("UPDATE plans SET status = ?, updated_at = ? WHERE id = ?",
                         (status, _now(), plan_id))

    def get(self, plan_id: str) -> dict | None:
        row = self.db.query_one("SELECT * FROM plans WHERE id = ?", (plan_id,))
        if row is None:
            return None
        steps = self.db.query(
            "SELECT * FROM plan_steps WHERE plan_id = ? ORDER BY step_index ASC", (plan_id,)
        )
        return {"plan": dict(row), "steps": [dict(s) for s in steps]}
