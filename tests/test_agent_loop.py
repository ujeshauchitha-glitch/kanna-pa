from __future__ import annotations

from core.agent.loop import AgentLoop
from core.agent.state import AgentState
from core.errors import PlanningError
from core.permissions.levels import PermissionLevel
from core.planner.plan import Plan, PlanStep
from core.tools.registry import ToolRegistry
from core.tools.result import ToolResult
from core.tools.schema import obj, string


class _AlwaysSucceedsTool:
    name = "test_always_succeeds"
    description = "always succeeds"
    permission = PermissionLevel.LOW
    input_schema = obj({})
    output_schema = obj({})

    def execute(self, args, ctx):
        return ToolResult.ok({})


class _FlakyTool:
    """Fails on its first N invocations, then succeeds — simulates a transient failure."""
    name = "test_flaky"
    description = "fails then succeeds"
    permission = PermissionLevel.LOW
    input_schema = obj({})
    output_schema = obj({})

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def execute(self, args, ctx):
        self.calls += 1
        if self.calls <= self.fail_times:
            return ToolResult.fail("transient_error", "not ready yet")
        return ToolResult.ok({})


class _AlwaysFailsTool:
    name = "test_always_fails"
    description = "always fails"
    permission = PermissionLevel.LOW
    input_schema = obj({})
    output_schema = obj({})

    def execute(self, args, ctx):
        return ToolResult.fail("permanent_error", "this will never work")


class _FixedPlanPlanner:
    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    def create_plan(self, request, registry):
        return self._plan


class _RaisingPlanner:
    def create_plan(self, request, registry):
        raise PlanningError("cannot understand this request")


def test_happy_path_completes(ctx):
    registry = ToolRegistry()
    registry.register(_AlwaysSucceedsTool())
    plan = Plan(request="do it", steps=[PlanStep(tool_name="test_always_succeeds", args={})])
    loop = AgentLoop(registry, _FixedPlanPlanner(plan), ctx, max_corrections=2)

    result = loop.run("do it")
    assert result.state == AgentState.COMPLETE
    assert result.outcomes[0].result.success


def test_transient_failure_recovers_via_correction(ctx):
    registry = ToolRegistry()
    flaky = _FlakyTool(fail_times=1)
    registry.register(flaky)
    plan = Plan(request="do it", steps=[PlanStep(tool_name="test_flaky", args={})])
    loop = AgentLoop(registry, _FixedPlanPlanner(plan), ctx, max_corrections=3)

    result = loop.run("do it")
    assert result.state == AgentState.COMPLETE
    assert result.outcomes[0].attempts == 2
    assert flaky.calls == 2


def test_permanent_failure_reports_failed_never_success(ctx):
    registry = ToolRegistry()
    registry.register(_AlwaysFailsTool())
    plan = Plan(request="do it", steps=[PlanStep(tool_name="test_always_fails", args={})])
    loop = AgentLoop(registry, _FixedPlanPlanner(plan), ctx, max_corrections=2)

    result = loop.run("do it")
    assert result.state == AgentState.FAILED
    assert "this will never work" in result.message
    assert result.outcomes[0].attempts == 3  # 1 initial + 2 corrections, budget exhausted


def test_exceeding_correction_budget_still_reports_failed_not_complete(ctx):
    registry = ToolRegistry()
    flaky = _FlakyTool(fail_times=10)  # never succeeds within budget
    registry.register(flaky)
    plan = Plan(request="do it", steps=[PlanStep(tool_name="test_flaky", args={})])
    loop = AgentLoop(registry, _FixedPlanPlanner(plan), ctx, max_corrections=2)

    result = loop.run("do it")
    assert result.state == AgentState.FAILED
    assert result.state != AgentState.COMPLETE


def test_unplannable_request_reports_blocked(ctx):
    registry = ToolRegistry()
    loop = AgentLoop(registry, _RaisingPlanner(), ctx)
    result = loop.run("do something incomprehensible")
    assert result.state == AgentState.BLOCKED
    assert "cannot understand" in result.message


def test_run_persists_plan_and_steps(ctx):
    registry = ToolRegistry()
    registry.register(_AlwaysSucceedsTool())
    plan = Plan(request="do it", steps=[PlanStep(tool_name="test_always_succeeds", args={})])
    loop = AgentLoop(registry, _FixedPlanPlanner(plan), ctx)
    loop.run("do it")

    rows = ctx.db.query("SELECT * FROM plans")
    assert len(rows) == 1
    assert rows[0]["status"] == "complete"

    step_rows = ctx.db.query("SELECT * FROM plan_steps")
    assert len(step_rows) == 1
    assert step_rows[0]["status"] == "ok"
