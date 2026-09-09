"""`kanna scheduler ...` subcommands."""
from __future__ import annotations

import argparse

from automation.scheduler.scheduler import Scheduler
from automation.scheduler.schedule import Schedule
from automation.scheduler.store import SchedulerStore
from core.bootstrap import Kanna


def register(subparsers: argparse._SubParsersAction) -> None:
    sched_parser = subparsers.add_parser("scheduler", help="Automation scheduler")
    sched_sub = sched_parser.add_subparsers(dest="scheduler_command", required=True)

    tick_p = sched_sub.add_parser("tick", help="Run every due schedule once")
    tick_p.set_defaults(func=_cmd_tick)

    list_p = sched_sub.add_parser("list", help="List schedules")
    list_p.set_defaults(func=_cmd_list)


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
