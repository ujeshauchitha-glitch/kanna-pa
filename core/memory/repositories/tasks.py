"""Repository for the task system (distinct from finance/plan tables)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from core.memory.db import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskRecord:
    id: str
    title: str
    description: str | None
    status: str
    project: str | None
    due_date: str | None
    created_at: str
    updated_at: str


_STATUSES = {"pending", "in_progress", "completed", "cancelled"}


class TaskRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, title: str, description: str | None = None, project: str | None = None,
               due_date: str | None = None) -> TaskRecord:
        task_id = str(uuid.uuid4())
        now = _now()
        self.db.execute(
            "INSERT INTO tasks (id, title, description, status, project, due_date, created_at, updated_at) "
            "VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)",
            (task_id, title, description, project, due_date, now, now),
        )
        return TaskRecord(id=task_id, title=title, description=description, status="pending",
                           project=project, due_date=due_date, created_at=now, updated_at=now)

    def set_status(self, task_id: str, status: str) -> None:
        if status not in _STATUSES:
            raise ValueError(f"invalid task status: {status}")
        self.db.execute("UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
                         (status, _now(), task_id))

    def get(self, task_id: str) -> TaskRecord | None:
        row = self.db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,))
        return None if row is None else self._from_row(row)

    def list(self, status: str | None = None) -> list[TaskRecord]:
        if status:
            rows = self.db.query("SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC", (status,))
        else:
            rows = self.db.query("SELECT * FROM tasks ORDER BY created_at DESC")
        return [self._from_row(r) for r in rows]

    @staticmethod
    def _from_row(row) -> TaskRecord:
        return TaskRecord(id=row["id"], title=row["title"], description=row["description"],
                           status=row["status"], project=row["project"], due_date=row["due_date"],
                           created_at=row["created_at"], updated_at=row["updated_at"])
