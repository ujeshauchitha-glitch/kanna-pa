from __future__ import annotations

import threading

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


class _SetsEventThenSucceedsTool:
    """Succeeds, but sets a shared event as a side effect — used to
    simulate cancellation being requested while this step was running."""
    name = "test_sets_event"
    description = "succeeds and sets an event"
    permission = PermissionLevel.LOW
    input_schema = obj({})
    output_schema = obj({})

    def __init__(self, event: threading.Event) -> None:
        self.event = event

    def execute(self, args, ctx):
        self.event.set()
        return ToolResult.ok({})


class _CancelingFlakyTool:
    """Fails every call, and requests cancellation after the first one —
    proves a correction-retry loop stops instead of retrying once the
    request arrives, without touching the attempt already in flight."""
    name = "test_canceling_flaky"
    description = "fails and requests cancellation on its first call"
    permission = PermissionLevel.LOW
    input_schema = obj({})
    output_schema = obj({})

    def __init__(self, cancel_event: threading.Event) -> None:
        self.cancel_event = cancel_event
        self.calls = 0

    def execute(self, args, ctx):
        self.calls += 1
        if self.calls == 1:
            self.cancel_event.set()
        return ToolResult.fail("transient_error", "not ready yet")


# -- Cooperative cancellation: checked between steps/retries, never mid-call --

def test_cancel_set_before_run_skips_every_step(ctx):
    registry = ToolRegistry()
    registry.register(_AlwaysSucceedsTool())
    plan = Plan(request="do it", steps=[PlanStep(tool_name="test_always_succeeds", args={})])
    cancel = threading.Event()
    cancel.set()
    loop = AgentLoop(registry, _FixedPlanPlanner(plan), ctx)

    result = loop.run("do it", cancel=cancel)

    assert result.state == AgentState.CANCELLED
    assert result.outcomes == []
    assert "0 of 1" in result.message
    assert ctx.db.query_one("SELECT status FROM plans")["status"] == "cancelled"


def test_cancel_between_steps_never_starts_the_next_one(ctx):
    registry = ToolRegistry()
    cancel = threading.Event()
    registry.register(_SetsEventThenSucceedsTool(cancel))
    registry.register(_AlwaysSucceedsTool())
    plan = Plan(request="do it", steps=[
        PlanStep(tool_name="test_sets_event", args={}),
        PlanStep(tool_name="test_always_succeeds", args={}),
    ])
    loop = AgentLoop(registry, _FixedPlanPlanner(plan), ctx)

    result = loop.run("do it", cancel=cancel)

    assert result.state == AgentState.CANCELLED
    # The first step ran to completion and its (real) success is kept —
    # cancellation never claims a completed side effect was undone.
    assert len(result.outcomes) == 1
    assert result.outcomes[0].result.success
    assert "1 of 2" in result.message
    assert ctx.db.query_one("SELECT status FROM plans")["status"] == "cancelled"


def test_cancel_stops_correction_retries_not_the_in_flight_attempt(ctx):
    registry = ToolRegistry()
    cancel = threading.Event()
    tool = _CancelingFlakyTool(cancel)
    registry.register(tool)
    plan = Plan(request="do it", steps=[PlanStep(tool_name="test_canceling_flaky", args={})])
    loop = AgentLoop(registry, _FixedPlanPlanner(plan), ctx, max_corrections=5)

    result = loop.run("do it", cancel=cancel)

    assert result.state == AgentState.CANCELLED
    assert tool.calls == 1  # the attempt already running finished normally; no retry started
    assert "not undone" in result.message


def test_no_cancel_argument_behaves_exactly_as_before(ctx):
    """Every existing caller that doesn't pass `cancel` must see identical
    behavior — cancellation is strictly additive."""
    registry = ToolRegistry()
    registry.register(_AlwaysSucceedsTool())
    plan = Plan(request="do it", steps=[PlanStep(tool_name="test_always_succeeds", args={})])
    loop = AgentLoop(registry, _FixedPlanPlanner(plan), ctx)

    result = loop.run("do it")  # no cancel kwarg at all

    assert result.state == AgentState.COMPLETE


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
