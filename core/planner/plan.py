"""The plan data structure the agent loop executes."""
from __future__ import annotations

from dataclasses import dataclass, field
from core.tools.schema import Schema, array, integer, obj, string

EXPECTED_SCHEMA = obj({
    "success": Schema(type="boolean", enum=(True,)),
    "min_files_created": integer(minimum=0),
    "min_files_modified": integer(minimum=0),
    "data_nonempty_key": string(),
    "data_equals": obj({}),
    "files_exist": array(string()),
    "exit_code": integer(),
})


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
