"""Kanna's command-line interface.

Every subcommand does real work through the same `core.bootstrap.Kanna`
wiring the agent loop uses — there is no separate "CLI-only" code path
for tools, finance, or the agent.
"""
from __future__ import annotations

import argparse
import sys

from core.bootstrap import bootstrap
from core.permissions.gate import CLIPromptGate
from interfaces.cli.commands import finance as finance_cmd
from interfaces.cli.commands import scheduler as scheduler_cmd
from interfaces.cli.commands import tasks as tasks_cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kanna", description="Kanna — a personal AI work-execution agent")
    parser.add_argument("--yes", action="store_true",
                         help="Interactively approve REVIEW-level actions instead of denying them")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Initialize Kanna's local database and config").set_defaults(
        func=_cmd_init)

    ask_p = subparsers.add_parser("ask", help="Give Kanna a natural-language request")
    ask_p.add_argument("request")
    ask_p.set_defaults(func=_cmd_ask)

    tools_p = subparsers.add_parser("tools", help="Inspect the tool registry")
    tools_sub = tools_p.add_subparsers(dest="tools_command", required=True)
    tools_sub.add_parser("list", help="List every registered tool").set_defaults(func=_cmd_tools_list)

    db_p = subparsers.add_parser("db", help="Database maintenance")
    db_sub = db_p.add_subparsers(dest="db_command", required=True)
    db_sub.add_parser("migrate", help="Apply pending migrations").set_defaults(func=_cmd_db_migrate)

    finance_cmd.register(subparsers)
    tasks_cmd.register(subparsers)
    scheduler_cmd.register(subparsers)

    return parser


def _cmd_init(args: argparse.Namespace, kanna) -> int:
    print("Kanna initialized.")
    print(f"  database: {kanna.db.path}")
    print(f"  sandbox roots: {', '.join(str(r) for r in kanna.sandbox.roots)}")
    print(f"  planner: {type(kanna.planner).__name__}")
    return 0


def _cmd_ask(args: argparse.Namespace, kanna) -> int:
    result = kanna.agent_loop().run(args.request)
    print(result.message)
    return 0 if result.state.value == "complete" else 1


def _cmd_tools_list(args: argparse.Namespace, kanna) -> int:
    for tool in kanna.registry.describe():
        print(f"{tool['name']:28} [{tool['permission']:6}] {tool['description']}")
    return 0


def _cmd_db_migrate(args: argparse.Namespace, kanna) -> int:
    applied = kanna.db.migrate()
    if applied:
        print(f"Applied migrations: {applied}")
    else:
        print("Database already up to date.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    gate = CLIPromptGate() if getattr(args, "yes", False) else None
    kanna = bootstrap(gate=gate)
    try:
        return args.func(args, kanna)
    finally:
        kanna.close()


if __name__ == "__main__":
    sys.exit(main())
