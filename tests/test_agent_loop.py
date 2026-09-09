from __future__ import annotations

from core.agent.loop import AgentLoop
from core.agent.state import AgentState
from core.errors import PlanningError
from core.permissions.levels import PermissionLevel
from core.planner.plan import Plan, PlanStep
from core.tools.registry import ToolRegistry
from core.tools.result import ToolResult
from core.tools.schema import boolean, obj


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


class _RevisingTool:
    """Fails until called with args={"fixed": True} — lets a test prove the
    args a revision produces are actually the ones that get invoked."""
    name = "test_revising"
    description = "fails unless fixed"
    permission = PermissionLevel.LOW
    input_schema = obj({"fixed": boolean()})
    output_schema = obj({})

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def execute(self, args, ctx):
        self.calls.append(args)
        if args.get("fixed") is True:
            return ToolResult.ok({})
        return ToolResult.fail("wrong_args", "not fixed yet")


class _FixedPlanPlannerWithRevision(_FixedPlanPlanner):
    """A `_FixedPlanPlanner` that also offers `revise_step`, scripted to
    return a fixed revised step regardless of the failure reason."""

    def __init__(self, plan: Plan, revised_step: PlanStep) -> None:
        super().__init__(plan)
        self._revised_step = revised_step
        self.revise_calls: list[tuple[PlanStep, list[str]]] = []

    def revise_step(self, step, reasons, registry):
        self.revise_calls.append((step, reasons))
        return self._revised_step


class _RaisingRevisionPlanner(_FixedPlanPlanner):
    """Offers `revise_step` but it always fails — proves `AgentLoop._revise`
    falls back to the unrevised step rather than crashing the run."""

    def revise_step(self, step, reasons, registry):
        raise PlanningError("could not revise")


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


# -- LLM-driven correction: AgentLoop asks the planner to revise args on failure --

def test_correction_uses_planner_revised_args_when_available(ctx):
    registry = ToolRegistry()
    tool = _RevisingTool()
    registry.register(tool)
    plan = Plan(request="do it", steps=[PlanStep(tool_name="test_revising", args={"fixed": False})])
    revised_step = PlanStep(tool_name="test_revising", args={"fixed": True})
    planner = _FixedPlanPlannerWithRevision(plan, revised_step)
    loop = AgentLoop(registry, planner, ctx, max_corrections=2)

    result = loop.run("do it")

    assert result.state == AgentState.COMPLETE
    # The revised args (not the original, wrong ones) are what got invoked
    # the second time — proof this is genuine args revision, not a blind retry.
    assert tool.calls == [{"fixed": False}, {"fixed": True}]
    assert len(planner.revise_calls) == 1
    assert planner.revise_calls[0][0].args == {"fixed": False}


def test_correction_falls_back_to_identical_retry_when_revision_fails(ctx):
    registry = ToolRegistry()
    flaky = _FlakyTool(fail_times=1)
    registry.register(flaky)
    plan = Plan(request="do it", steps=[PlanStep(tool_name="test_flaky", args={})])
    planner = _RaisingRevisionPlanner(plan)
    loop = AgentLoop(registry, planner, ctx, max_corrections=3)

    result = loop.run("do it")

    # revise_step raised PlanningError every time; the loop still recovers
    # by retrying with the unchanged args, exactly like a planner with no
    # revise_step capability at all.
    assert result.state == AgentState.COMPLETE
    assert flaky.calls == 2


def test_correction_without_revision_capability_retries_identically(ctx):
    """RuleBasedPlanner-shaped planners (no revise_step) keep today's
    behavior: same args, every attempt."""
    registry = ToolRegistry()
    flaky = _FlakyTool(fail_times=1)
    registry.register(flaky)
    plan = Plan(request="do it", steps=[PlanStep(tool_name="test_flaky", args={})])
    loop = AgentLoop(registry, _FixedPlanPlanner(plan), ctx, max_corrections=3)

    result = loop.run("do it")

    assert result.state == AgentState.COMPLETE
    assert flaky.calls == 2
