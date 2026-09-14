"""Approval gates.

A `Decision.REQUIRE_APPROVAL` from the policy still has to be resolved by
something before the action runs. An `ApprovalGate` is that something.
The default (`DenyAllGate`) never lets a REVIEW-level action through on
its own — a human or an explicit pre-approval has to grant it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from core.permissions.levels import PermissionLevel
from core.permissions.trust import rule_matches

if TYPE_CHECKING:
    from core.memory.repositories.trust_rules import TrustRuleRepository


class ApprovalGate(Protocol):
    def approve(self, *, tool_name: str, args: dict[str, Any], level: PermissionLevel,
                reason: str) -> bool:
        """Return True if the action may proceed."""
        ...


class DenyAllGate:
    """Never approves anything. Safe default for unattended/automated runs."""

    def approve(self, *, tool_name: str, args: dict[str, Any], level: PermissionLevel,
                reason: str) -> bool:
        return False


class CLIPromptGate:
    """Prompts on stdin/stdout. Suitable for an interactive CLI session."""

    def approve(self, *, tool_name: str, args: dict[str, Any], level: PermissionLevel,
                reason: str) -> bool:
        print(f"\n[APPROVAL REQUIRED] {tool_name} ({level.name}): {reason}")
        print(f"  args: {args}")
        answer = input("  Allow this action? [y/N] ").strip().lower()
        return answer in ("y", "yes")


class PreApprovedGate:
    """Approves only actions matching a caller-supplied allowlist of tool names.

    This is the seam for "trusted automations" mentioned in the spec:
    a scheduler or workflow can construct one of these with a fixed set
    of tool names it's been explicitly configured to trust, without
    granting blanket approval to everything.
    """

    def __init__(self, allowed_tool_names: set[str]) -> None:
        self._allowed = set(allowed_tool_names)

    def approve(self, *, tool_name: str, args: dict[str, Any], level: PermissionLevel,
                reason: str) -> bool:
        return tool_name in self._allowed


class TrustStoreGate:
    """Consults persisted standing approvals (`TrustRuleRepository`) before
    falling back to another gate.

    This is the "trusted automation" configuration surface: a user
    grants standing approval to a specific tool+argument pattern once
    (`kanna trust add ...`), and every future matching call — an
    interactive `kanna ask`, a scheduler-triggered run, or any other
    agent-loop invocation, since they all go through the same
    `bootstrap()` wiring — auto-approves without asking again.
    `PreApprovedGate` above is the in-memory, programmatic version of
    the same idea (a caller-supplied set of tool names, no persistence,
    no args matching); this is its durable, user-configurable
    counterpart, matched on args too, not just tool name.

    `fallback` (typically `DenyAllGate` for unattended runs, or
    `CLIPromptGate` for an interactive session) still decides anything
    the trust store doesn't cover — this only *adds* a way to opt
    specific, named actions out of asking every time; it never removes
    the existing safety behavior for everything else.
    """

    def __init__(self, repository: TrustRuleRepository, fallback: ApprovalGate) -> None:
        self._repository = repository
        self._fallback = fallback

    def approve(self, *, tool_name: str, args: dict[str, Any], level: PermissionLevel,
                reason: str) -> bool:
        for rule in self._repository.list_all():
            if rule_matches(rule_tool_name=rule.tool_name, rule_args_pattern=rule.args_pattern,
                             call_tool_name=tool_name, call_args=args):
                return True
        return self._fallback.approve(tool_name=tool_name, args=args, level=level, reason=reason)
