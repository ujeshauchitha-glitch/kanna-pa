"""The `Tool` protocol every capability implements.

Deliberately a small surface: a name, a description, an input/output
`Schema`, a `PermissionLevel`, and `execute()`. Anything with this shape
can be registered and invoked uniformly, whether it's a filesystem
operation, a finance action, or (later) a browser or computer-control
action.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.result import ToolResult
from core.tools.schema import Schema


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    input_schema: Schema
    output_schema: Schema
    permission: PermissionLevel

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        ...
