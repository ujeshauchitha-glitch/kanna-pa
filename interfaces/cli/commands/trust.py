"""`kanna trust ...` — manage standing tool-approval rules ("trusted automation").

A trust rule auto-approves a REVIEW-level tool call matching a
tool name + args pattern, without asking again — see
`core.permissions.gate.TrustStoreGate` for how it's consulted, and
`core.memory.repositories.trust_rules` for the underlying storage. The
rule table itself is the audit trail of what's been granted: every row
carries when (`created_at`) and, if given, why (`note`).
"""
from __future__ import annotations

import argparse
import sys

from core.bootstrap import Kanna


def register(subparsers: argparse._SubParsersAction) -> None:
    trust_parser = subparsers.add_parser(
        "trust", help="Manage standing tool-approval rules (trusted automation)")
    trust_sub = trust_parser.add_subparsers(dest="trust_command", required=True)

    trust_sub.add_parser("list", help="List every standing approval rule").set_defaults(
        func=_cmd_list)

    add_p = trust_sub.add_parser(
        "add", help="Grant standing approval to a tool (optionally scoped to specific args)")
    add_p.add_argument("tool_name", help="Exact tool name, e.g. 'computer_click'")
    add_p.add_argument(
        "--arg", action="append", default=[], metavar="KEY=VALUE",
        help="Restrict the grant to calls where this arg matches exactly (repeatable). "
             "Omit entirely to trust every call to this tool, regardless of args.")
    add_p.add_argument("--note", default="", help="Why this is trusted, for the audit trail")
    add_p.set_defaults(func=_cmd_add)

    remove_p = trust_sub.add_parser("remove", help="Revoke a standing approval rule")
    remove_p.add_argument("id", type=int, help="Rule id, from 'kanna trust list'")
    remove_p.set_defaults(func=_cmd_remove)


def _parse_args_pattern(raw: list[str]) -> dict[str, str] | None:
    """Turn ["key=value", ...] into {"key": "value", ...}, or None on a malformed entry."""
    pattern: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            return None
        key, _, value = item.partition("=")
        pattern[key] = value
    return pattern


def _cmd_list(args: argparse.Namespace, kanna: Kanna) -> int:
    rules = kanna.trust_rules.list_all()
    if not rules:
        print("No standing approval rules.")
        return 0
    for rule in rules:
        scope = "any args" if not rule.args_pattern else str(rule.args_pattern)
        note = f" — {rule.note}" if rule.note else ""
        print(f"[{rule.id}] {rule.tool_name} ({scope}){note}  (granted {rule.created_at})")
    return 0


def _cmd_add(args: argparse.Namespace, kanna: Kanna) -> int:
    if not kanna.registry.has(args.tool_name):
        print(f"error: no tool registered as '{args.tool_name}' — see 'kanna tools list'",
              file=sys.stderr)
        return 1

    pattern = _parse_args_pattern(args.arg)
    if pattern is None:
        print("error: --arg must be KEY=VALUE", file=sys.stderr)
        return 1

    rule = kanna.trust_rules.add(tool_name=args.tool_name, args_pattern=pattern, note=args.note)
    print(f"Granted standing approval [{rule.id}]: {rule.tool_name} "
          f"({'any args' if not pattern else pattern}).")
    return 0


def _cmd_remove(args: argparse.Namespace, kanna: Kanna) -> int:
    if kanna.trust_rules.remove(args.id):
        print(f"Revoked rule [{args.id}].")
        return 0
    print(f"error: no rule with id {args.id}", file=sys.stderr)
    return 1
