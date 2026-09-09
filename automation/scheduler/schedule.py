"""Schedule variants and pure due-time computation.

Kept free of any I/O so `due()`/`compute_next_run()` are trivially unit
testable — `SchedulerStore` (persistence) and `Scheduler` (the `tick`
command) are the only things that touch the database or the clock.

All datetimes in this module are naive but conceptually UTC — schedule
configs store plain ISO strings ("2024-03-01T09:00:00", no offset), and
callers are expected to pass a naive UTC `now` (see `Scheduler.tick`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta

_WEEKLY_REQUIRED = {"weekday", "time", "anchor_date"}


@dataclass
class Schedule:
    id: str
    name: str
    kind: str  # "once" | "interval" | "weekly"
    config: dict
    active: bool = True
    next_run_at: str | None = None  # ISO 8601 UTC; maintained by the store

    def __post_init__(self) -> None:
        if self.kind not in ("once", "interval", "weekly"):
            raise ValueError(f"unknown schedule kind: {self.kind!r}")
        if self.kind == "once" and "run_at" not in self.config:
            raise ValueError("'once' schedules require config.run_at")
        if self.kind == "interval" and "seconds" not in self.config:
            raise ValueError("'interval' schedules require config.seconds")
        if self.kind == "weekly" and not _WEEKLY_REQUIRED.issubset(self.config):
            raise ValueError(f"'weekly' schedules require config keys: {_WEEKLY_REQUIRED}")


def compute_next_run(schedule: Schedule, *, now: datetime, last_run: datetime | None) -> datetime | None:
    """Return the next UTC datetime this schedule should fire at, or None if it never will again."""
    if schedule.kind == "once":
        run_at = datetime.fromisoformat(schedule.config["run_at"])
        return None if last_run is not None else run_at

    if schedule.kind == "interval":
        seconds = int(schedule.config["seconds"])
        anchor = datetime.fromisoformat(schedule.config.get("anchor", now.isoformat()))
        if last_run is None:
            return anchor
        return last_run + timedelta(seconds=seconds)

    if schedule.kind == "weekly":
        return _next_weekly(schedule, now=now, last_run=last_run)

    raise AssertionError("unreachable")  # pragma: no cover


def _next_weekly(schedule: Schedule, *, now: datetime, last_run: datetime | None) -> datetime:
    weekday = int(schedule.config["weekday"])  # 0=Monday .. 6=Sunday
    hour, minute = (int(p) for p in schedule.config["time"].split(":"))
    every_n_weeks = int(schedule.config.get("every_n_weeks", 1))
    anchor_date = datetime.fromisoformat(schedule.config["anchor_date"]).date()

    search_start = (last_run.date() + timedelta(days=1)) if last_run else now.date()

    candidate = search_start
    for _ in range(every_n_weeks * 7 + 8):  # bounded search, always terminates
        if candidate.weekday() == weekday:
            weeks_since_anchor = (candidate - anchor_date).days // 7
            if weeks_since_anchor >= 0 and weeks_since_anchor % every_n_weeks == 0:
                return datetime.combine(candidate, time(hour=hour, minute=minute))
        candidate += timedelta(days=1)

    raise RuntimeError("could not compute next weekly occurrence")  # pragma: no cover


def is_due(schedule: Schedule, *, now: datetime, last_run: datetime | None) -> bool:
    if not schedule.active:
        return False
    next_run = compute_next_run(schedule, now=now, last_run=last_run)
    return next_run is not None and next_run <= now
