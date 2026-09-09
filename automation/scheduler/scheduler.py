"""The scheduler's `tick`: find due schedules and run them.

Phase 1 has no daemon — `kanna scheduler tick` (see
`interfaces/cli/commands/scheduler.py`) is meant to be invoked by the
OS's own cron/systemd-timer, or manually. Each due schedule is handed to
an `on_due` callback (typically "feed this schedule's request into the
agent loop"); if none is supplied, being due is still recorded as a job
so `list_due()`-style inspection isn't silently lossy.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from automation.scheduler.schedule import Schedule, compute_next_run, is_due
from automation.scheduler.store import SchedulerStore

OnDue = Callable[[Schedule], dict]


class Scheduler:
    def __init__(self, store: SchedulerStore, *, on_due: OnDue | None = None) -> None:
        self.store = store
        self.on_due = on_due

    def tick(self, *, now: datetime | None = None) -> list[dict]:
        """Run every due schedule once. Returns a list of {schedule_id, name, status} outcomes.

        All schedule times are naive datetimes conceptually in UTC (see
        `schedule.py`) — `now` defaults to the current UTC time with
        `tzinfo` stripped so it compares against them without raising.
        """
        now = now or datetime.now(timezone.utc).replace(tzinfo=None)
        outcomes = []

        for schedule in self.store.list(active_only=True):
            last_run = self.store.last_run_at(schedule.id)
            if not is_due(schedule, now=now, last_run=last_run):
                continue

            if self.on_due is not None:
                try:
                    result = self.on_due(schedule)
                    status = "succeeded"
                except Exception as exc:  # noqa: BLE001 - one bad job must not kill the tick
                    result = {"error": str(exc)}
                    status = "failed"
            else:
                result = {"note": "no executor configured; recorded as due"}
                status = "skipped"

            self.store.record_job(schedule.id, status=status, result=result)
            next_run = compute_next_run(schedule, now=now, last_run=now)
            self.store.set_next_run_at(schedule.id, next_run)
            if next_run is None:
                self.store.set_active(schedule.id, False)

            outcomes.append({"schedule_id": schedule.id, "name": schedule.name, "status": status})

        return outcomes
