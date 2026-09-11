from __future__ import annotations

from datetime import datetime, timezone

from automation.scheduler.daemon import SchedulerDaemon, install_signal_handlers
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


def test_interval_job_survives_multiple_ticks_with_persisted_timestamp(db):
    store = SchedulerStore(db)
    schedule = store.create("repeat", "interval", {"seconds": 60, "anchor": "2024-01-01T00:00:00"})
    scheduler = Scheduler(store, on_due=lambda s: {"state": "complete"})
    assert len(scheduler.tick(now=_dt("2024-01-01T00:00:00+00:00"))) == 1
    assert store.last_run_at(schedule.id) == _dt("2024-01-01T00:00:00")
    assert scheduler.tick(now=_dt("2024-01-01T00:00:30")) == []
    assert len(scheduler.tick(now=_dt("2024-01-01T00:01:00"))) == 1


def test_legacy_aware_job_timestamp_normalized(db):
    store = SchedulerStore(db)
    schedule = store.create("repeat", "interval", {"seconds": 60})
    store.record_job(schedule.id, status="succeeded", result={},
                     ran_at=_dt("2024-01-01T05:30:00+05:30"))
    assert store.last_run_at(schedule.id) == _dt("2024-01-01T00:00:00")


# -- SchedulerDaemon: the "set and forget" run loop --

def test_daemon_run_n_ticks_calls_tick_the_right_number_of_times(db):
    store = SchedulerStore(db)
    tick_count = {"n": 0}
    scheduler = Scheduler(store, on_due=lambda s: None)
    # Patch tick() to just count calls — this test is about the daemon's
    # loop mechanics, not scheduler.tick()'s own behavior (covered above).
    scheduler.tick = lambda: (tick_count.__setitem__("n", tick_count["n"] + 1), [])[1]

    sleeps: list[float] = []
    daemon = SchedulerDaemon(scheduler, interval_seconds=5, sleep=sleeps.append)
    daemon.run_n_ticks(3)

    assert tick_count["n"] == 3
    # Sleeps between ticks, not after the last one.
    assert sleeps == [5, 5]


def test_daemon_run_n_ticks_reports_each_ticks_outcomes_via_callback(db):
    store = SchedulerStore(db)
    # Far in the past, so it's already due whenever this test actually runs
    # — no need to fake `now` to get a deterministic first-tick result.
    store.create("reminder", "once", {"run_at": "2000-01-01T00:00:00"})
    scheduler = Scheduler(store, on_due=lambda s: {"ok": True})

    reported: list[list[dict]] = []
    daemon = SchedulerDaemon(scheduler, sleep=lambda _: None, on_tick=reported.append)
    daemon.run_n_ticks(2)

    assert len(reported) == 2
    assert reported[0][0]["status"] == "succeeded"  # first tick: the schedule was due
    assert reported[1] == []  # second tick: 'once' already fired, nothing due


def test_daemon_run_forever_stops_when_stop_is_called(db):
    store = SchedulerStore(db)
    scheduler = Scheduler(store, on_due=lambda s: None)

    calls = {"ticks": 0}
    daemon = SchedulerDaemon(scheduler, sleep=lambda _: None)

    def _stop_after_three_ticks(outcomes):
        calls["ticks"] += 1
        if calls["ticks"] >= 3:
            daemon.stop()

    daemon._on_tick = _stop_after_three_ticks
    daemon.run_forever()

    assert calls["ticks"] == 3


def test_install_signal_handlers_sigint_calls_stop(db):
    import signal

    store = SchedulerStore(db)
    scheduler = Scheduler(store, on_due=lambda s: None)
    daemon = SchedulerDaemon(scheduler)
    install_signal_handlers(daemon)

    assert daemon._stop is False
    os_kill_self_with_sigint = signal.getsignal(signal.SIGINT)
    os_kill_self_with_sigint(signal.SIGINT, None)  # simulate delivery without actually raising
    assert daemon._stop is True

    # Restore the default handler so this test doesn't leak state to others.
    signal.signal(signal.SIGINT, signal.default_int_handler)
