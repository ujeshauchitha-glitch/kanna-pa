"""`kanna scheduler ...` subcommands.

`add` is what actually populates the scheduler — Phase 1 shipped
`SchedulerStore.create()` with nothing in the CLI calling it, so a
schedule could only be created from Python. `daemon` is the "set and
forget" mode (see `automation/scheduler/daemon.py` and
`docs/SCHEDULER.md`) — `tick` still exists for cron/systemd-timer-driven
invocation, unaffected.
"""
from __future__ import annotations

import argparse
import sys

from automation.scheduler.daemon import SchedulerDaemon, install_signal_handlers
from automation.scheduler.scheduler import Scheduler
from automation.scheduler.schedule import Schedule
from automation.scheduler.store import SchedulerStore
from core.bootstrap import Kanna


def register(subparsers: argparse._SubParsersAction) -> None:
    sched_parser = subparsers.add_parser("scheduler", help="Automation scheduler")
    sched_sub = sched_parser.add_subparsers(dest="scheduler_command", required=True)

    tick_p = sched_sub.add_parser("tick", help="Run every due schedule once")
    tick_p.set_defaults(func=_cmd_tick)

    daemon_p = sched_sub.add_parser(
        "daemon", help="Run the scheduler continuously (foreground) instead of via cron/tick")
    daemon_p.add_argument("--interval-seconds", type=float, default=60.0,
                           help="How often to check for due schedules (default: 60)")
    daemon_p.add_argument("--ticks", type=int, default=None,
                           help="Run exactly N ticks and exit, instead of running forever "
                                "(mainly for testing/manual runs)")
    daemon_p.set_defaults(func=_cmd_daemon)

    list_p = sched_sub.add_parser("list", help="List schedules")
    list_p.set_defaults(func=_cmd_list)

    add_p = sched_sub.add_parser("add", help="Create a schedule")
    add_p.add_argument("name")
    add_p.add_argument("request", help="The natural-language request to run when this fires")
    add_p.add_argument("--kind", required=True, choices=["once", "interval", "weekly"])
    add_p.add_argument("--run-at", help="ISO datetime, e.g. 2024-03-01T09:00:00 — required for 'once'")
    add_p.add_argument("--seconds", type=int, help="Interval in seconds — required for 'interval'")
    add_p.add_argument("--anchor", help="ISO datetime the interval is anchored to (default: now)")
    add_p.add_argument("--weekday", type=int, help="0=Monday..6=Sunday — required for 'weekly'")
    add_p.add_argument("--time", dest="time_of_day", metavar="HH:MM",
                        help="Time of day — required for 'weekly'")
    add_p.add_argument("--anchor-date", help="ISO date the weekly cadence is anchored to — "
                                              "required for 'weekly'")
    add_p.add_argument("--every-n-weeks", type=int, default=1,
                        help="For 'weekly': fire every N weeks instead of every week (default: 1)")
    add_p.set_defaults(func=_cmd_add)

    remove_p = sched_sub.add_parser(
        "remove", help="Deactivate a schedule (soft-remove — history is kept, see 'kanna scheduler list')")
    remove_p.add_argument("id")
    remove_p.set_defaults(func=_cmd_remove)


def _on_due(schedule: Schedule, kanna: Kanna) -> dict:
    request = schedule.config.get("request")
    if not request:
        return {"note": "schedule has no 'request' in config; nothing to run"}
    result = kanna.agent_loop().run(request)
    return {"state": result.state.value, "message": result.message}


def _cmd_tick(args: argparse.Namespace, kanna: Kanna) -> int:
    store = SchedulerStore(kanna.db)
    scheduler = Scheduler(store, on_due=lambda s: _on_due(s, kanna))
    outcomes = scheduler.tick()
    if not outcomes:
        print("No schedules due.")
        return 0
    for outcome in outcomes:
        print(f"{outcome['name']} ({outcome['schedule_id']}): {outcome['status']}")
    return 0


def _print_tick_outcomes(outcomes: list[dict]) -> None:
    if not outcomes:
        print("tick: no schedules due")
        return
    for outcome in outcomes:
        print(f"tick: {outcome['name']} ({outcome['schedule_id']}): {outcome['status']}")


def _cmd_daemon(args: argparse.Namespace, kanna: Kanna) -> int:
    store = SchedulerStore(kanna.db)
    scheduler = Scheduler(store, on_due=lambda s: _on_due(s, kanna))
    daemon = SchedulerDaemon(scheduler, interval_seconds=args.interval_seconds,
                              on_tick=_print_tick_outcomes)

    if args.ticks is not None:
        daemon.run_n_ticks(args.ticks)
        return 0

    print(f"Scheduler daemon running (checking every {args.interval_seconds:.0f}s). "
          "Ctrl-C to stop.")
    install_signal_handlers(daemon)
    daemon.run_forever()
    print("Scheduler daemon stopped.")
    return 0


def _cmd_list(args: argparse.Namespace, kanna: Kanna) -> int:
    store = SchedulerStore(kanna.db)
    schedules = store.list()
    if not schedules:
        print("No schedules configured.")
        return 0
    for s in schedules:
        state = "active" if s.active else "inactive"
        print(f"[{state:8}] {s.id}  {s.name}  ({s.kind})  next_run_at={s.next_run_at}")
    return 0


def _cmd_add(args: argparse.Namespace, kanna: Kanna) -> int:
    config: dict = {"request": args.request}

    if args.kind == "once":
        if not args.run_at:
            print("error: --run-at is required for --kind once", file=sys.stderr)
            return 1
        config["run_at"] = args.run_at
    elif args.kind == "interval":
        if args.seconds is None:
            print("error: --seconds is required for --kind interval", file=sys.stderr)
            return 1
        config["seconds"] = args.seconds
        if args.anchor:
            config["anchor"] = args.anchor
    elif args.kind == "weekly":
        missing = [flag for flag, val in (
            ("--weekday", args.weekday), ("--time", args.time_of_day),
            ("--anchor-date", args.anchor_date),
        ) if val is None]
        if missing:
            print(f"error: {', '.join(missing)} required for --kind weekly", file=sys.stderr)
            return 1
        config.update(weekday=args.weekday, time=args.time_of_day, anchor_date=args.anchor_date,
                       every_n_weeks=args.every_n_weeks)

    try:
        schedule = SchedulerStore(kanna.db).create(args.name, args.kind, config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Created schedule [{schedule.id}] {schedule.name} ({schedule.kind}).")
    return 0


def _cmd_remove(args: argparse.Namespace, kanna: Kanna) -> int:
    store = SchedulerStore(kanna.db)
    if store.get(args.id) is None:
        print(f"error: no schedule with id {args.id}", file=sys.stderr)
        return 1
    store.set_active(args.id, False)
    print(f"Deactivated schedule [{args.id}].")
    return 0
