"""Postcondition checking.

A plan step passes or fails by *code* inspecting the `ToolResult`, never
by asking an LLM whether it "looks right". Each step's `expected` dict
names which checks apply; `verify()` runs exactly those and returns every
reason it failed (empty list = passed).
"""
from __future__ import annotations

from core.planner.plan import PlanStep
from core.tools.result import ToolResult


def verify(step: PlanStep, result: ToolResult) -> list[str]:
    """Return a list of failure reasons; empty means the step's postconditions held."""
    reasons: list[str] = []
    expected = step.expected or {"success": True}

    if expected.get("success", True) and not result.success:
        err = result.error
        detail = f": {err.code} — {err.message}" if err else ""
        reasons.append(f"tool call did not succeed{detail}")

    min_created = expected.get("min_files_created")
    if min_created is not None and len(result.files_created) < min_created:
        reasons.append(f"expected at least {min_created} file(s) created, got {len(result.files_created)}")

    min_modified = expected.get("min_files_modified")
    if min_modified is not None and len(result.files_modified) < min_modified:
        reasons.append(f"expected at least {min_modified} file(s) modified, got {len(result.files_modified)}")

    nonempty_key = expected.get("data_nonempty_key")
    if nonempty_key is not None and not result.data.get(nonempty_key):
        reasons.append(f"expected non-empty '{nonempty_key}' in result data")

    return reasons
