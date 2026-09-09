"""Pure matching logic for trust rules — "does this rule cover this call?"

Kept separate from persistence (`core.memory.repositories.trust_rules`)
and from the gate that consults it at approval time
(`core.permissions.gate.TrustStoreGate`) so the actual matching rule is
independently unit-tested as a pure function, the same discipline
`core.agent.verifier.verify()` and `core.permissions.policy.
PermissionPolicy.decide()` already follow.
"""
from __future__ import annotations

from typing import Any


def rule_matches(*, rule_tool_name: str, rule_args_pattern: dict[str, Any],
                  call_tool_name: str, call_args: dict[str, Any]) -> bool:
    """True if a trust rule (tool name + an args pattern) covers this call.

    `rule_args_pattern` is a *subset* match: every key it specifies must
    equal the same key in the actual call's args — a missing key or a
    different value means no match. An empty pattern matches any args
    for that tool (a blanket per-tool trust). Extra keys in `call_args`
    the pattern doesn't mention are ignored; the pattern only constrains
    what it explicitly names, so a rule like `{"name": "firefox"}` for
    `computer_open_application` matches that one app regardless of what
    other args a future version of the tool might add.
    """
    if rule_tool_name != call_tool_name:
        return False
    return all(call_args.get(key) == value for key, value in rule_args_pattern.items())
