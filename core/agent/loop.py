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

Correction is *args revision*, not blind retry, when the planner
supports it: before each retry, `_revise()` asks the planner (via the
optional `revise_step()` capability — see `core.planner.llm_planner.
LLMPlanner.revise_step`) to fix the failing step's args given the
specific failure reason, still re-verified in code exactly the same way
afterward. A planner without that capability (`RuleBasedPlanner`, or an
`LLMPlanner` whose revision attempt itself fails) falls back to retrying
with the same args unchanged — today's behavior, never regressed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from copy import deepcopy
import threading

from core.agent.state import AgentState
from core.agent.verifier import verify
from core.errors import PlanningError
from core.events.types import Event
from core.memory.repositories.plans import PlanRepository
from core.planner.base import Planner
from core.planner.plan import Plan, PlanStep
from core.planner.bindings import resolve, validate_plan
from core.tools.context import ToolContext
from core.tools.registry import ToolRegistry
from core.tools.result import ToolResult


@dataclass
class StepOutcome:
    step: PlanStep
    result: ToolResult
    attempts: int
    history: list[dict] = field(default_factory=list)


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

    def run(self, request: str, *, cancel: threading.Event | None = None) -> AgentRunResult:
        self._emit(AgentState.UNDERSTANDING, request=request)

        self._emit(AgentState.PLANNING)
        try:
            plan = self.planner.create_plan(request, self.registry)
            validate_plan(plan, self.registry)
        except PlanningError as exc:
            self._emit(AgentState.BLOCKED, reason=str(exc))
            return AgentRunResult(state=AgentState.BLOCKED, plan=None, message=str(exc))

        self._emit(AgentState.SELECTING, tool_count=len(plan.steps))
        plan_id = self._plan_repo.create(plan, session_id=self.ctx.session_id)

        outcomes: list[StepOutcome] = []
        for index, step in enumerate(plan.steps):
            # Checked here — before the next tool invocation or retry,
            # never mid-call — so a step already in flight always runs to
            # a real outcome; only work that hasn't started yet is
            # actually skipped. See docs/DESKTOP.md's cancellation section.
            if cancel is not None and cancel.is_set():
                return self._cancelled(plan_id, plan, outcomes)
            try:
                resolved_step = replace(step, args=resolve(
                    step.args, [o.result.to_dict() for o in outcomes]))
                outcome = self._run_step(resolved_step, cancel=cancel)
            except PlanningError as exc:
                outcome = StepOutcome(step, ToolResult.fail("binding_failed", str(exc)), 0)
            outcomes.append(outcome)
            self._plan_repo.record_step_result(
                plan_id, index,
                status="ok" if outcome.result.success else "failed",
                result={**outcome.result.to_dict(), "attempts": outcome.history,
                        "resolved_args": outcome.step.args},
            )

            if not outcome.result.success:
                if cancel is not None and cancel.is_set():
                    return self._cancelled(plan_id, plan, outcomes)
                self._plan_repo.set_status(plan_id, "failed")
                reasons = verify(outcome.step, outcome.result, self.ctx)
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

    def _cancelled(self, plan_id: str, plan: Plan, outcomes: list[StepOutcome]) -> AgentRunResult:
        # Never claims a side effect already performed was undone — it
        # wasn't, and this says so rather than staying silent about it.
        self._plan_repo.set_status(plan_id, "cancelled")
        self._emit(AgentState.CANCELLED, completed_steps=len(outcomes))
        done = sum(1 for o in outcomes if o.result.success)
        message = (f"Cancelled after {done} of {len(plan.steps)} step(s) completed. "
                   "Actions already taken were not undone.")
        return AgentRunResult(state=AgentState.CANCELLED, plan=plan, outcomes=outcomes, message=message)

    def _run_step(self, step: PlanStep, *, cancel: threading.Event | None = None) -> StepOutcome:
        current = step
        history = []
        terminal_errors = {"approval_denied", "permission_denied", "sandbox_violation",
                           "invalid_input", "invalid_output"}
        for attempt in range(1, max(0, self.max_corrections) + 2):
            self._emit(AgentState.EXECUTING, tool=current.tool_name)
            result = self.registry.invoke(current.tool_name, deepcopy(current.args), self.ctx)
            self._emit(AgentState.OBSERVING, tool=current.tool_name, success=result.success)
            self._emit(AgentState.CHECKING, tool=current.tool_name)
            reasons = verify(current, result, self.ctx)
            history.append({"attempt": attempt, "args": deepcopy(current.args),
                            "result": deepcopy(result.to_dict()), "verification_errors": reasons})
            if not reasons:
                break
            # Never replay a successful side effect to fix a failed check. Never
            # ask a model to route around an approval or sandbox denial.
            if result.success or (result.error and (
                    result.error.code in terminal_errors or result.error.code.endswith("_unavailable"))):
                break
            # Checked before the next retry, never mid-call: the attempt
            # that already ran is reported as-is; no further retry starts.
            if cancel is not None and cancel.is_set():
                self._emit(AgentState.CANCELLED, tool=current.tool_name, attempt=attempt)
                break
            if attempt <= self.max_corrections:
                current = self._revise(current, reasons)
                self._emit(AgentState.CORRECTING, tool=current.tool_name, attempt=attempt,
                           reasons=reasons)
        self._emit(AgentState.VERIFYING, tool=current.tool_name, passed=not reasons)
        if reasons and result.success:
            result = ToolResult.fail("verification_failed", "; ".join(reasons),
                                     data=result.data, files_created=result.files_created,
                                     files_modified=result.files_modified,
                                     files_deleted=result.files_deleted, metadata=result.metadata,
                                     duration_ms=result.duration_ms)
        return StepOutcome(step=current, result=result, attempts=len(history), history=history)

    def _revise(self, step: PlanStep, reasons: list[str]) -> PlanStep:
        """Before a correction retry, ask the planner to fix the step's args
        if it can — see the module docstring. Never raises: any revision
        failure (no capability, LLM unavailable, malformed response,
        invalid args) falls back to the original step unchanged, which is
        exactly what ran before this capability existed.
        """
        revise = getattr(self.planner, "revise_step", None)
        if revise is None:
            return step
        try:
            revised = revise(deepcopy(step), reasons, self.registry)
            # Optional planners may revise arguments only, never weaken checks.
            return replace(step, args=revised.args)
        except PlanningError:
            return step

    @staticmethod
    def _summarize(outcomes: list[StepOutcome]) -> str:
        if not outcomes:
            return "Nothing to do."
        lines = []
        for outcome in outcomes:
            desc = outcome.step.description or outcome.step.tool_name
            lines.append(f"- {desc}: done")
            data = outcome.result.data
            if isinstance(data.get("message"), str):
                lines.append(data["message"])
            for path in outcome.result.files_created + outcome.result.files_modified:
                lines.append(f"  Output: {path}")
        return "Completed:\n" + "\n".join(lines)

    def _emit(self, state: AgentState, **payload) -> None:
        self.ctx.event_bus.publish(Event(
            topic=f"agent.state.{state.value}", payload=payload, session_id=self.ctx.session_id,
        ))
