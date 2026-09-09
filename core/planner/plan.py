"""The plan data structure the agent loop executes."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlanStep:
    tool_name: str
    args: dict
    description: str = ""
    # Postconditions the verifier checks after execution, e.g.
    # {"success": True} (default) or {"success": True, "min_files_created": 1}.
    expected: dict = field(default_factory=lambda: {"success": True})


@dataclass
class Plan:
    request: str
    steps: list[PlanStep]
    rationale: str = ""
