"""Typed, backward-only result references for sequential plans; no evaluation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import re

from core.errors import PlanningError, ValidationError
from core.tools.schema import Schema

_REFERENCE = re.compile(r"(0|[1-9][0-9]*)\.(data|files_created|files_modified|metadata)(\.[A-Za-z0-9_]+)*\Z")


def reference_parts(value: dict, index: int) -> list[str]:
    ref = value.get("$ref")
    if set(value) != {"$ref"} or not isinstance(ref, str) or not _REFERENCE.fullmatch(ref):
        raise PlanningError("reference must be exactly {'$ref': '<step-index>.data.<key>'}")
    parts = ref.split(".")
    if int(parts[0]) >= index:
        raise PlanningError(f"reference {ref!r} must target an earlier step")
    return parts


def validate_template(value, schema: Schema | None, index: int) -> None:
    """Validate literal structure now; referenced values are validated at invocation."""
    if isinstance(value, dict) and "$ref" in value:
        reference_parts(value, index)
        return
    if schema is not None:
        replace(schema, properties={}, items=None).validate(value)
    if isinstance(value, dict):
        for key, item in value.items():
            validate_template(item, schema.properties.get(key) if schema else None, index)
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_template(item, schema.items if schema else None, index)


def resolve(value, results: list[dict]):
    """Copy values so downstream tools cannot mutate prior evidence or the plan."""
    if isinstance(value, dict):
        if "$ref" in value:
            parts = reference_parts(value, len(results))
            current = results[int(parts[0])]
            try:
                for part in parts[1:]:
                    if isinstance(current, dict):
                        current = current[part]
                    elif isinstance(current, list) and part.isdecimal():
                        current = current[int(part)]
                    else:
                        raise KeyError(part)
            except (KeyError, IndexError) as exc:
                raise PlanningError(f"unresolved result reference: {value['$ref']}") from exc
            return deepcopy(current)
        return {key: resolve(item, results) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [resolve(item, results) for item in value]
    return deepcopy(value)


def validate_plan(plan, registry) -> None:
    from core.planner.plan import EXPECTED_SCHEMA

    if not plan.steps:
        raise PlanningError("plan has no executable steps")
    for index, step in enumerate(plan.steps):
        if not isinstance(step.tool_name, str) or not registry.has(step.tool_name):
            raise PlanningError(f"step {index}: unknown tool {step.tool_name!r}")
        if not isinstance(step.args, dict):
            raise PlanningError(f"step {index}: args must be an object")
        try:
            validate_template(step.args, registry.get(step.tool_name).input_schema, index)
            EXPECTED_SCHEMA.validate(step.expected)
            unknown = set(step.expected) - set(EXPECTED_SCHEMA.properties)
            if unknown:
                raise PlanningError(f"unknown postconditions: {sorted(unknown)}")
        except ValidationError as exc:
            raise PlanningError(f"step {index}: {exc}") from exc
