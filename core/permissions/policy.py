"""The permission policy: decides what happens for a given tool call.

Rules are data (a list of `Rule`), not code branches, so adding a
"trusted automation" later is a matter of appending a rule rather than
touching this module's logic.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.permissions.levels import Decision, PermissionLevel

logger = logging.getLogger("kanna.permissions.policy")


@dataclass
class Rule:
    """Matches a tool call and overrides the default decision for it.

    `predicate` receives (tool_name, args) and returns True if the rule
    applies. `decision` is what to return when it matches.
    """

    name: str
    predicate: Callable[[str, dict[str, Any]], bool]
    decision: Decision


class PermissionPolicy:
    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules: list[Rule] = rules or []

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    def decide(self, *, tool_name: str, args: dict[str, Any], level: PermissionLevel) -> Decision:
        for rule in self.rules:
            if rule.predicate(tool_name, args):
                logger.debug("policy rule '%s' matched for %s -> %s", rule.name, tool_name, rule.decision)
                return rule.decision

        if level == PermissionLevel.LOW:
            return Decision.ALLOW
        if level == PermissionLevel.REVIEW:
            return Decision.REQUIRE_APPROVAL
        return Decision.DENY  # RESTRICTED: no default path to yes
