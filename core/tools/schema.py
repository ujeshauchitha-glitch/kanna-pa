"""A minimal hand-rolled JSON-Schema subset.

Two things need a description of a tool's inputs/outputs: our own runtime
validator (so a tool never runs with malformed args) and the Anthropic
tool-use API (which wants a JSON-Schema `dict`). One `Schema` object
serves both — `validate()` for us, `to_json_schema()` for the LLM — so the
two never drift apart.

Supports the subset Kanna actually needs: object/array/string/number/
integer/boolean/null, `required`, `properties`, `items`, `enum`, `default`,
`minimum`/`maximum`, `pattern`. That's deliberately not the full spec.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core.errors import ValidationError

_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list, tuple),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}


@dataclass
class Schema:
    type: str
    properties: dict[str, "Schema"] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    items: "Schema | None" = None
    enum: tuple[Any, ...] | None = None
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    pattern: str | None = None
    description: str = ""

    def to_json_schema(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.description:
            out["description"] = self.description
        if self.type == "object":
            out["properties"] = {k: v.to_json_schema() for k, v in self.properties.items()}
            if self.required:
                out["required"] = list(self.required)
        if self.type == "array" and self.items is not None:
            out["items"] = self.items.to_json_schema()
        if self.enum is not None:
            out["enum"] = list(self.enum)
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.maximum is not None:
            out["maximum"] = self.maximum
        if self.pattern is not None:
            out["pattern"] = self.pattern
        if self.default is not None:
            out["default"] = self.default
        return out

    def validate(self, value: Any, *, path: str = "$") -> None:
        """Raise `ValidationError` if `value` does not conform to this schema."""
        expected_types = _TYPE_MAP[self.type]
        # bool is a subclass of int in Python; keep integer/number strict.
        if self.type in ("integer", "number") and isinstance(value, bool):
            raise ValidationError(f"{path}: expected {self.type}, got boolean")
        if not isinstance(value, expected_types):
            raise ValidationError(f"{path}: expected {self.type}, got {type(value).__name__}")

        if self.enum is not None and value not in self.enum:
            raise ValidationError(f"{path}: {value!r} is not one of {list(self.enum)}")

        if self.type in ("integer", "number"):
            if self.minimum is not None and value < self.minimum:
                raise ValidationError(f"{path}: {value} is below minimum {self.minimum}")
            if self.maximum is not None and value > self.maximum:
                raise ValidationError(f"{path}: {value} exceeds maximum {self.maximum}")

        if self.type == "string" and self.pattern is not None:
            if not re.match(self.pattern, value):
                raise ValidationError(f"{path}: {value!r} does not match pattern {self.pattern}")

        if self.type == "object":
            for key in self.required:
                if key not in value:
                    raise ValidationError(f"{path}: missing required property '{key}'")
            for key, val in value.items():
                sub_schema = self.properties.get(key)
                if sub_schema is not None:
                    sub_schema.validate(val, path=f"{path}.{key}")

        if self.type == "array" and self.items is not None:
            for i, item in enumerate(value):
                self.items.validate(item, path=f"{path}[{i}]")


def obj(properties: dict[str, Schema], required: tuple[str, ...] = (), description: str = "") -> Schema:
    return Schema(type="object", properties=properties, required=required, description=description)


def string(*, description: str = "", enum: tuple[str, ...] | None = None, pattern: str | None = None,
           default: Any = None) -> Schema:
    return Schema(type="string", description=description, enum=enum, pattern=pattern, default=default)


def integer(*, description: str = "", minimum: float | None = None, maximum: float | None = None,
            default: Any = None) -> Schema:
    return Schema(type="integer", description=description, minimum=minimum, maximum=maximum, default=default)


def number(*, description: str = "", minimum: float | None = None, maximum: float | None = None) -> Schema:
    return Schema(type="number", description=description, minimum=minimum, maximum=maximum)


def boolean(*, description: str = "", default: Any = None) -> Schema:
    return Schema(type="boolean", description=description, default=default)


def array(items: Schema, *, description: str = "") -> Schema:
    return Schema(type="array", items=items, description=description)
