"""SQLite persistence for schedules and their run history."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from automation.scheduler.schedule import Schedule
from core.memory.db import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SchedulerStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, name: str, kind: str, config: dict, *, active: bool = True) -> Schedule:
        schedule = Schedule(id=str(uuid.uuid4()), name=name, kind=kind, config=config, active=active)
        now = _now()
        self.db.execute(
            "INSERT INTO schedules (id, name, schedule_type, config, next_run_at, active, "
            " created_at, updated_at) VALUES (?, ?, ?, ?, NULL, ?, ?, ?)",
            (schedule.id, name, kind, json.dumps(config), 1 if active else 0, now, now),
        )
        return schedule

    def list(self, *, active_only: bool = False) -> list[Schedule]:
        if active_only:
            rows = self.db.query("SELECT * FROM schedules WHERE active = 1")
        else:
            rows = self.db.query("SELECT * FROM schedules")
        return [self._from_row(r) for r in rows]

    def get(self, schedule_id: str) -> Schedule | None:
        row = self.db.query_one("SELECT * FROM schedules WHERE id = ?", (schedule_id,))
        return None if row is None else self._from_row(row)

    def set_next_run_at(self, schedule_id: str, next_run_at: datetime | None) -> None:
        self.db.execute(
            "UPDATE schedules SET next_run_at = ?, updated_at = ? WHERE id = ?",
            (next_run_at.isoformat() if next_run_at else None, _now(), schedule_id),
        )

    def set_active(self, schedule_id: str, active: bool) -> None:
        self.db.execute("UPDATE schedules SET active = ?, updated_at = ? WHERE id = ?",
                         (1 if active else 0, _now(), schedule_id))

    def last_run_at(self, schedule_id: str) -> datetime | None:
        row = self.db.query_one(
            "SELECT ran_at FROM jobs WHERE schedule_id = ? ORDER BY ran_at DESC LIMIT 1",
            (schedule_id,),
        )
        return None if row is None else datetime.fromisoformat(row["ran_at"])

    def record_job(self, schedule_id: str, *, status: str, result: dict) -> None:
        self.db.execute(
            "INSERT INTO jobs (id, schedule_id, status, ran_at, result) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), schedule_id, status, _now(), json.dumps(result)),
        )

    @staticmethod
    def _from_row(row) -> Schedule:
        return Schedule(id=row["id"], name=row["name"], kind=row["schedule_type"],
                         config=json.loads(row["config"]), active=bool(row["active"]),
                         next_run_at=row["next_run_at"])
