"""`kanna task ...` subcommands."""
from __future__ import annotations

import argparse

from core.bootstrap import Kanna
from core.tasks.service import TaskService


def register(subparsers: argparse._SubParsersAction) -> None:
    task_parser = subparsers.add_parser("task", help="Track tasks")
    task_sub = task_parser.add_subparsers(dest="task_command", required=True)

    add_p = task_sub.add_parser("add", help="Create a task")
    add_p.add_argument("title")
    add_p.add_argument("--description", default=None)
    add_p.add_argument("--project", default=None)
    add_p.add_argument("--due-date", default=None)
    add_p.set_defaults(func=_cmd_add)

    list_p = task_sub.add_parser("list", help="List tasks")
    list_p.add_argument("--status", default=None,
                         choices=["pending", "in_progress", "completed", "cancelled"])
    list_p.set_defaults(func=_cmd_list)

    for verb in ("start", "complete", "cancel"):
        p = task_sub.add_parser(verb, help=f"Mark a task as {verb}")
        p.add_argument("task_id")
        p.set_defaults(func=_cmd_transition, _verb=verb)


def _cmd_add(args: argparse.Namespace, kanna: Kanna) -> int:
    task = TaskService(kanna.db).create(args.title, description=args.description,
                                         project=args.project, due_date=args.due_date)
    print(f"Created task {task.id}: {task.title}")
    return 0


def _cmd_list(args: argparse.Namespace, kanna: Kanna) -> int:
    tasks = TaskService(kanna.db).list(status=args.status)
    if not tasks:
        print("No tasks.")
        return 0
    for task in tasks:
        print(f"[{task.status:11}] {task.id}  {task.title}")
    return 0


def _cmd_transition(args: argparse.Namespace, kanna: Kanna) -> int:
    svc = TaskService(kanna.db)
    getattr(svc, args._verb)(args.task_id)
    print(f"Task {args.task_id} -> {args._verb}")
    return 0
