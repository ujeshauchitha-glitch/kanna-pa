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
from core.tools.registry import ToolRegistry

_SYSTEM_TEMPLATE = """You are Kanna's planner. Given a user request and a list of available tools, \
respond with ONLY a JSON object (no prose, no markdown fences) of the form:

{{"rationale": "<one sentence>", "steps": [{{"tool_name": "<tool>", "args": {{...}}, \
"description": "<what this step does>"}}]}}

Only use tool names from this list, and make sure "args" satisfies each tool's input schema:

{tool_catalog}

If the request cannot be fulfilled with these tools, respond with {{"rationale": "...", "steps": []}}.
"""

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

        steps_raw = payload.get("steps", [])
        if not steps_raw:
            raise PlanningError("LLM reported the request cannot be fulfilled with available tools")

        steps: list[PlanStep] = []
        for i, step in enumerate(steps_raw):
            tool_name = step.get("tool_name")
            args = step.get("args", {})
            if not registry.has(tool_name):
                raise PlanningError(f"step {i}: LLM referenced unknown tool '{tool_name}'")
            try:
                registry.get(tool_name).input_schema.validate(args)
            except ValidationError as exc:
                raise PlanningError(f"step {i}: args for '{tool_name}' failed validation: {exc}") from exc
            steps.append(PlanStep(tool_name=tool_name, args=args, description=step.get("description", "")))

        return Plan(request=request, steps=steps, rationale=payload.get("rationale", ""))
