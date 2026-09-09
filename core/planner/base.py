"""The `Planner` protocol."""
from __future__ import annotations

from typing import Protocol

from core.planner.plan import Plan
from core.tools.registry import ToolRegistry


class Planner(Protocol):
    def create_plan(self, request: str, registry: ToolRegistry) -> Plan:
        """Turn a natural-language request into an ordered list of tool calls.

        Raises `PlanningError` if no plan can be produced.
        """
        ...
