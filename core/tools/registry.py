"""The tool registry: where every tool is registered and invoked from.

`invoke()` is the single choke point every tool call passes through:
validate input -> permission check (policy + gate) -> execute -> validate
output -> log to `execution_log` -> publish an event. Tools are never
called directly by the agent loop or CLI — always through here — so this
is the one place that enforces the contract.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from core.errors import PermissionDenied, ToolExecutionError, ToolNotFound, ValidationError
from core.events.types import Event
from core.memory.repositories.execution_log import ExecutionLogRepository
from core.permissions.gate import ApprovalGate, DenyAllGate
from core.permissions.levels import Decision
from core.permissions.policy import PermissionPolicy
from core.tools.context import ToolContext
from core.tools.protocol import Tool
from core.tools.result import ToolResult


@dataclass
class ToolRegistry:
    policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    gate: ApprovalGate = field(default_factory=DenyAllGate)
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"a tool named '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFound(f"no tool registered as '{name}'") from None

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict]:
        """Tool descriptions in a shape suitable for the planner / LLM tool-use API."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema.to_json_schema(),
                "permission": tool.permission.name,
            }
            for tool in self._tools.values()
        ]

    def invoke(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        tool = self.get(name)
        start = time.monotonic()

        try:
            tool.input_schema.validate(args)
        except ValidationError as exc:
            result = ToolResult.fail("invalid_input", str(exc))
            self._finish(tool, args, ctx, result, start)
            return result

        decision = self.policy.decide(tool_name=tool.name, args=args, level=tool.permission)
        if decision == Decision.DENY:
            result = ToolResult.fail("permission_denied", f"'{tool.name}' is denied by policy")
            self._finish(tool, args, ctx, result, start)
            return result
        if decision == Decision.REQUIRE_APPROVAL:
            approved = self.gate.approve(
                tool_name=tool.name, args=args, level=tool.permission,
                reason=f"{tool.name} requires review-level approval",
            )
            if not approved:
                result = ToolResult.fail(
                    "approval_denied", f"'{tool.name}' requires approval, which was not granted"
                )
                self._finish(tool, args, ctx, result, start)
                return result

        try:
            result = tool.execute(args, ctx)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: a tool must never crash the agent
            result = ToolResult.fail("execution_error", str(exc), details={"exception_type": type(exc).__name__})
            self._finish(tool, args, ctx, result, start)
            return result

        if result.success:
            try:
                tool.output_schema.validate(result.data)
            except ValidationError as exc:
                result = ToolResult.fail("invalid_output", f"tool returned malformed data: {exc}")

        self._finish(tool, args, ctx, result, start)
        return result

    def _finish(self, tool: Tool, args: dict, ctx: ToolContext, result: ToolResult, start: float) -> None:
        result.duration_ms = (time.monotonic() - start) * 1000
        try:
            ExecutionLogRepository(ctx.db).record(
                session_id=ctx.session_id, tool_name=tool.name, args=args, result=result
            )
        except Exception:  # noqa: BLE001 - logging must never break tool execution
            ctx.logger.exception("failed to persist execution_log entry for %s", tool.name)
        ctx.event_bus.publish(Event(
            topic="tool.executed",
            payload={"tool_name": tool.name, "success": result.success, "duration_ms": result.duration_ms},
            session_id=ctx.session_id,
        ))
