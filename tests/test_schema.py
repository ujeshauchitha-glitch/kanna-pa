from __future__ import annotations

import pytest

from core.errors import ValidationError
from core.tools.schema import array, boolean, integer, obj, string


def test_string_type_and_enum():
    s = string(enum=("a", "b"))
    s.validate("a")
    with pytest.raises(ValidationError):
        s.validate("c")
    with pytest.raises(ValidationError):
        s.validate(1)


def test_integer_bool_is_not_integer():
    s = integer()
    with pytest.raises(ValidationError):
        s.validate(True)  # bool is an int subclass in Python; schema must reject it


def test_integer_bounds():
    s = integer(minimum=0, maximum=10)
    s.validate(5)
    with pytest.raises(ValidationError):
        s.validate(-1)
    with pytest.raises(ValidationError):
        s.validate(11)


def test_object_required_and_nested():
    schema = obj({"name": string(), "age": integer()}, required=("name",))
    schema.validate({"name": "a"})
    schema.validate({"name": "a", "age": 5})
    with pytest.raises(ValidationError):
        schema.validate({"age": 5})
    with pytest.raises(ValidationError):
        schema.validate({"name": "a", "age": "not an int"})


def test_array_items():
    schema = array(integer())
    schema.validate([1, 2, 3])
    with pytest.raises(ValidationError):
        schema.validate([1, "two"])


def test_pattern():
    schema = string(pattern=r"^\d{4}-\d{2}-\d{2}$")
    schema.validate("2024-01-01")
    with pytest.raises(ValidationError):
        schema.validate("not-a-date")


def test_to_json_schema_roundtrip_shape():
    schema = obj({"path": string(description="p")}, required=("path",))
    js = schema.to_json_schema()
    assert js["type"] == "object"
    assert js["required"] == ["path"]
    assert js["properties"]["path"]["type"] == "string"


def test_boolean_type():
    b = boolean()
    b.validate(True)
    with pytest.raises(ValidationError):
        b.validate("true")
