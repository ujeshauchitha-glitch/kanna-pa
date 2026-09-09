"""The agent loop.

Implements the core cycle the whole project is built around:

    UNDERSTAND -> PLAN -> SELECT TOOLS -> EXECUTE -> OBSERVE -> CHECK
        -> CORRECT IF NECESSARY -> VERIFY -> COMPLETE

"Understand"/"Plan"/"Select tools" collapse into one `planner.create_plan()`
call in Phase 1 (the planner both interprets the request and picks tools).
Each step is then executed, its result checked by `verifier.verify()`
(code, not a model's opinion), and a failing step is retried up to
`max_corrections` times before the whole run reports FAILED — with the
real blocker, never a fabricated success. Every state transition and
step outcome is persisted via `PlanRepository` so a run can be inspected
after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.agent.state import AgentState
from core.agent.verifier import verify
from core.errors import PlanningError
from core.events.types import Event
from core.memory.repositories.plans import PlanRepository
from core.planner.base import Planner
from core.planner.plan import Plan, PlanStep
from core.tools.context import ToolContext
from core.tools.registry import ToolRegistry
from core.tools.result import ToolResult


@dataclass
class StepOutcome:
    step: PlanStep
    result: ToolResult
    attempts: int


@dataclass
class AgentRunResult:
    state: AgentState
    plan: Plan | None
    outcomes: list[StepOutcome] = field(default_factory=list)
    message: str = ""


class AgentLoop:
    def __init__(self, registry: ToolRegistry, planner: Planner, ctx: ToolContext,
                 *, max_corrections: int = 3) -> None:
        self.registry = registry
        self.planner = planner
        self.ctx = ctx
        self.max_corrections = max_corrections
        self._plan_repo = PlanRepository(ctx.db)

    def run(self, request: str) -> AgentRunResult:
        self._emit(AgentState.UNDERSTANDING, request=request)

        self._emit(AgentState.PLANNING)
        try:
            plan = self.planner.create_plan(request, self.registry)
        except PlanningError as exc:
            self._emit(AgentState.BLOCKED, reason=str(exc))
            return AgentRunResult(state=AgentState.BLOCKED, plan=None, message=str(exc))

        self._emit(AgentState.SELECTING, tool_count=len(plan.steps))
        plan_id = self._plan_repo.create(plan, session_id=self.ctx.session_id)

        outcomes: list[StepOutcome] = []
        for index, step in enumerate(plan.steps):
            outcome = self._run_step(step)
            outcomes.append(outcome)
            self._plan_repo.record_step_result(
                plan_id, index,
                status="ok" if outcome.result.success else "failed",
                result=outcome.result.to_dict(),
            )

            if not outcome.result.success:
                self._plan_repo.set_status(plan_id, "failed")
                reasons = verify(step, outcome.result)
                blocker = "; ".join(reasons) if reasons else (
                    outcome.result.error.message if outcome.result.error else "unknown failure"
                )
                self._emit(AgentState.FAILED, step=step.tool_name, blocker=blocker)
                return AgentRunResult(
                    state=AgentState.FAILED, plan=plan, outcomes=outcomes,
                    message=f"Step '{step.tool_name}' failed after {outcome.attempts} attempt(s): {blocker}",
                )

        self._plan_repo.set_status(plan_id, "complete")
        self._emit(AgentState.COMPLETE)
        return AgentRunResult(
            state=AgentState.COMPLETE, plan=plan, outcomes=outcomes,
            message=self._summarize(outcomes),
        )

    def _run_step(self, step: PlanStep) -> StepOutcome:
        self._emit(AgentState.EXECUTING, tool=step.tool_name)
        result = self.registry.invoke(step.tool_name, step.args, self.ctx)
        attempts = 1

        self._emit(AgentState.OBSERVING, tool=step.tool_name, success=result.success)
        self._emit(AgentState.CHECKING, tool=step.tool_name)
        reasons = verify(step, result)

        while reasons and attempts <= self.max_corrections:
            self._emit(AgentState.CORRECTING, tool=step.tool_name, attempt=attempts,
                        reasons=reasons)
            result = self.registry.invoke(step.tool_name, step.args, self.ctx)
            attempts += 1
            reasons = verify(step, result)

        self._emit(AgentState.VERIFYING, tool=step.tool_name, passed=not reasons)
        return StepOutcome(step=step, result=result, attempts=attempts)

    @staticmethod
    def _summarize(outcomes: list[StepOutcome]) -> str:
        if not outcomes:
            return "Nothing to do."
        lines = []
        for outcome in outcomes:
            desc = outcome.step.description or outcome.step.tool_name
            lines.append(f"- {desc}: done")
        return "Completed:\n" + "\n".join(lines)

    def _emit(self, state: AgentState, **payload) -> None:
        self.ctx.event_bus.publish(Event(
            topic=f"agent.state.{state.value}", payload=payload, session_id=self.ctx.session_id,
        ))
