from __future__ import annotations

from datetime import datetime, timezone

from automation.scheduler.schedule import Schedule, compute_next_run, is_due
from automation.scheduler.scheduler import Scheduler
from automation.scheduler.store import SchedulerStore


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def test_once_schedule_due_before_first_run():
    schedule = Schedule(id="s1", name="test", kind="once", config={"run_at": "2024-03-01T09:00:00"})
    assert is_due(schedule, now=_dt("2024-03-01T10:00:00"), last_run=None)
    assert not is_due(schedule, now=_dt("2024-02-28T10:00:00"), last_run=None)


def test_once_schedule_never_due_again_after_running():
    schedule = Schedule(id="s1", name="test", kind="once", config={"run_at": "2024-03-01T09:00:00"})
    next_run = compute_next_run(schedule, now=_dt("2024-03-02T00:00:00"), last_run=_dt("2024-03-01T09:00:00"))
    assert next_run is None


def test_interval_schedule():
    schedule = Schedule(id="s1", name="test", kind="interval",
                         config={"seconds": 3600, "anchor": "2024-03-01T09:00:00"})
    first = compute_next_run(schedule, now=_dt("2024-03-01T08:00:00"), last_run=None)
    assert first == _dt("2024-03-01T09:00:00")

    second = compute_next_run(schedule, now=_dt("2024-03-01T09:30:00"),
                               last_run=_dt("2024-03-01T09:00:00"))
    assert second == _dt("2024-03-01T10:00:00")


def test_weekly_schedule_every_two_weeks_on_tuesday():
    # Anchor Tuesday: 2024-03-05 is a Tuesday.
    schedule = Schedule(id="s1", name="reminder", kind="weekly",
                         config={"weekday": 1, "time": "09:00", "every_n_weeks": 2,
                                 "anchor_date": "2024-03-05"})

    first = compute_next_run(schedule, now=_dt("2024-03-01T00:00:00"), last_run=None)
    assert first == _dt("2024-03-05T09:00:00")

    # The following Tuesday (2024-03-12) is skipped — every 2 weeks means the next
    # occurrence after the anchor's own week is 2024-03-19.
    second = compute_next_run(schedule, now=_dt("2024-03-06T00:00:00"),
                               last_run=_dt("2024-03-05T09:00:00"))
    assert second == _dt("2024-03-19T09:00:00")


def test_inactive_schedule_never_due():
    schedule = Schedule(id="s1", name="test", kind="once", config={"run_at": "2024-01-01T00:00:00"},
                         active=False)
    assert not is_due(schedule, now=_dt("2024-06-01T00:00:00"), last_run=None)


def test_scheduler_tick_runs_due_schedules_and_records_job(db):
    store = SchedulerStore(db)
    store.create("Environmental Sciences reminder", "once", {"run_at": "2024-03-01T09:00:00"})

    ran = []
    scheduler = Scheduler(store, on_due=lambda s: (ran.append(s.name), {"ok": True})[1])
    outcomes = scheduler.tick(now=datetime(2024, 3, 1, 10, 0, tzinfo=None))

    assert ran == ["Environmental Sciences reminder"]
    assert outcomes[0]["status"] == "succeeded"

    # A 'once' schedule is deactivated after it fires.
    assert store.list(active_only=True) == []


def test_scheduler_tick_without_executor_records_skipped(db):
    store = SchedulerStore(db)
    store.create("no-op", "once", {"run_at": "2024-01-01T00:00:00"})
    scheduler = Scheduler(store)
    outcomes = scheduler.tick(now=datetime(2024, 6, 1, 0, 0, tzinfo=None))
    assert outcomes[0]["status"] == "skipped"


def test_scheduler_tick_captures_executor_failure(db):
    store = SchedulerStore(db)
    store.create("broken", "once", {"run_at": "2024-01-01T00:00:00"})

    def _boom(schedule):
        raise RuntimeError("nope")

    scheduler = Scheduler(store, on_due=_boom)
    outcomes = scheduler.tick(now=datetime(2024, 6, 1, 0, 0, tzinfo=None))
    assert outcomes[0]["status"] == "failed"
