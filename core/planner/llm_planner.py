"""A planner backed by an `LLMProvider`.

Asks the model to return a JSON plan restricted to tools the registry
actually has, validates every step's args against that tool's input
schema before accepting the plan, and falls back to another planner
(normally `RuleBasedPlanner`) if the LLM is unavailable or returns
something that doesn't validate. The model never gets to invent a tool
name or skip validation — a malformed "plan" is not a plan.
"""
from __future__ import annotations

import json
import re

from core.errors import LLMUnavailable, PlanningError, ValidationError
from core.llm.base import LLMProvider, Message
from core.planner.plan import Plan, PlanStep
from core.planner.bindings import validate_plan
from core.tools.registry import ToolRegistry

_SYSTEM_TEMPLATE = """You are Kanna's planner. Given a user request and a list of available tools, \
respond with ONLY a JSON object (no prose, no markdown fences) of the form:

{{"rationale": "<one sentence>", "steps": [{{"tool_name": "<tool>", "args": {{...}}, \
"description": "<what this step does>"}}]}}

Only use tool names from this list, and make sure "args" satisfies each tool's input schema:

{tool_catalog}

Use result references for values learned by earlier steps, including nested arrays/objects:
{{"$ref": "0.data.content"}} copies the content from step 0's actual result.
References use zero-based earlier step indices, then data/files_created/files_modified/metadata,
then dot-separated keys or list indices. They replace a whole value, never part of a string.
Do not invent file contents or outputs that have not been read. References are resolved and
validated against the tool schema before permission checks and execution.
Steps may include "expected": {{"success": true, "data_equals": {{"exists": true}},
"files_exist": ["report.pdf"]}}. Other checks: min_files_created, min_files_modified,
data_nonempty_key, exit_code. process_run defaults to requiring exit_code 0.
To re-render an extracted document, pass its title and sections by reference to a document
generation tool. A reference copies content; it cannot summarize or author new answers.

If the request cannot be fulfilled with these tools, respond with {{"rationale": "...", "steps": []}}.
"""

_REVISE_SYSTEM_TEMPLATE = """You are Kanna's planner, fixing one failing step of a plan already in \
progress. Given the tool's input schema, the args that were tried, and why the attempt failed, \
respond with ONLY a JSON object (no prose, no markdown fences) of the form:

{{"args": {{...}}}}

"args" must satisfy this tool's input schema exactly:

{tool_schema}

Keep every part of "args" that isn't implicated by the failure reason unchanged. Do not invent \
information you don't have (a file path, an amount, a name) to fill a gap — if the failure reason \
doesn't point to a fixable mistake in the args themselves (e.g. the tool is genuinely unavailable, or \
a value is missing that only the user could supply), return "args_tried" unchanged rather than \
guessing."""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMPlanner:
    def __init__(self, provider: LLMProvider, fallback=None) -> None:
        self.provider = provider
        self.fallback = fallback

    def create_plan(self, request: str, registry: ToolRegistry) -> Plan:
        try:
            return self._create_plan_via_llm(request, registry)
        except (LLMUnavailable, PlanningError):
            if self.fallback is not None:
                return self.fallback.create_plan(request, registry)
            raise

    def _create_plan_via_llm(self, request: str, registry: ToolRegistry) -> Plan:
        catalog = json.dumps(registry.describe(), indent=2)
        system = _SYSTEM_TEMPLATE.format(tool_catalog=catalog)

        response = self.provider.complete(
            [Message(role="user", content=request)], system=system,
        )

        match = _JSON_BLOCK_RE.search(response.content)
        if not match:
            raise PlanningError(f"LLM did not return a JSON plan: {response.content[:200]!r}")

        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise PlanningError(f"LLM returned invalid JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise PlanningError("LLM plan must be an object")
        steps_raw = payload.get("steps", [])
        if not isinstance(steps_raw, list) or not steps_raw:
            raise PlanningError("LLM reported the request cannot be fulfilled with available tools")

        steps: list[PlanStep] = []
        for i, step in enumerate(steps_raw):
            if not isinstance(step, dict):
                raise PlanningError(f"step {i}: must be an object")
            tool_name = step.get("tool_name")
            args = step.get("args", {})
            steps.append(PlanStep(tool_name=tool_name, args=args, description=step.get("description", ""),
                                  expected=step.get("expected", {"success": True})))

        plan = Plan(request=request, steps=steps, rationale=payload.get("rationale", ""))
        validate_plan(plan, registry)
        return plan

    def revise_step(self, step: PlanStep, reasons: list[str], registry: ToolRegistry) -> PlanStep:
        """Ask the LLM to fix a failing step's args, given why it failed.

        Only `args` can change — `tool_name`, `description`, and
        `expected` all carry over unchanged, so a revision can never
        smuggle in a different (unverified) tool call under the same
        step; the caller still re-runs `verify()` on whatever this
        produces, exactly as it would on an identical retry.

        This is an *extra* capability beyond the `Planner` protocol
        (`AgentLoop._run_step` checks for it with `hasattr` rather than
        every planner having to implement it) — `RuleBasedPlanner` has
        no LLM to ask, so it has no `revise_step`, and a failing step
        under it retries with the same args unchanged, same as before
        this existed.

        Raises `PlanningError` on any failure (LLM unavailable, no JSON
        in the response, args that don't validate) — the caller falls
        back to retrying with the original args unchanged rather than
        ever surfacing a revision failure as a crash. A correction
        attempt that doesn't work is not worse than no correction.
        """
        if not registry.has(step.tool_name):
            raise PlanningError(f"cannot revise: unknown tool {step.tool_name!r}")
        tool = registry.get(step.tool_name)

        system = _REVISE_SYSTEM_TEMPLATE.format(
            tool_schema=json.dumps(tool.input_schema.to_json_schema(), indent=2)
        )
        user_content = json.dumps({
            "tool_name": step.tool_name, "args_tried": step.args, "failure_reasons": reasons,
        }, indent=2)

        try:
            response = self.provider.complete([Message(role="user", content=user_content)], system=system)
        except LLMUnavailable as exc:
            raise PlanningError(f"cannot revise: LLM unavailable: {exc}") from exc

        match = _JSON_BLOCK_RE.search(response.content)
        if not match:
            raise PlanningError(f"LLM did not return a JSON revision: {response.content[:200]!r}")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise PlanningError(f"LLM returned invalid JSON revision: {exc}") from exc

        if not isinstance(payload, dict):
            raise PlanningError("LLM revision must be an object")
        args = payload.get("args")
        if not isinstance(args, dict):
            raise PlanningError("LLM revision response had no 'args' object")
        try:
            tool.input_schema.validate(args)
        except ValidationError as exc:
            raise PlanningError(f"revised args for {step.tool_name!r} failed validation: {exc}") from exc

        return PlanStep(tool_name=step.tool_name, args=args, description=step.description,
                         expected=step.expected)
