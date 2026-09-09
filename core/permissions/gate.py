"""Approval gates.

A `Decision.REQUIRE_APPROVAL` from the policy still has to be resolved by
something before the action runs. An `ApprovalGate` is that something.
The default (`DenyAllGate`) never lets a REVIEW-level action through on
its own — a human or an explicit pre-approval has to grant it.
"""
from __future__ import annotations

from typing import Any, Protocol

from core.permissions.levels import PermissionLevel


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
