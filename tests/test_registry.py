from __future__ import annotations

import pytest

from core.errors import ToolNotFound
from core.permissions.gate import DenyAllGate
from core.permissions.levels import PermissionLevel
from core.permissions.policy import PermissionPolicy
from core.tools.registry import ToolRegistry
from core.tools.result import ToolResult
from core.tools.schema import obj, string


class _EchoTool:
    name = "test_echo"
    description = "Echoes input"
    permission = PermissionLevel.LOW
    input_schema = obj({"text": string()}, required=("text",))
    output_schema = obj({"text": string()})

    def execute(self, args, ctx):
        return ToolResult.ok({"text": args["text"]})


class _CrashingTool:
    name = "test_crash"
    description = "Always raises"
    permission = PermissionLevel.LOW
    input_schema = obj({})
    output_schema = obj({})

    def execute(self, args, ctx):
        raise RuntimeError("boom")


class _BadOutputTool:
    name = "test_bad_output"
    description = "Returns output that fails its own schema"
    permission = PermissionLevel.LOW
    input_schema = obj({})
    output_schema = obj({"n": string()})

    def execute(self, args, ctx):
        return ToolResult.ok({"n": 123})  # violates output_schema (expects string)


def test_register_and_get():
    reg = ToolRegistry()
    reg.register(_EchoTool())
    assert reg.has("test_echo")
    assert reg.get("test_echo").name == "test_echo"


def test_register_duplicate_raises():
    reg = ToolRegistry()
    reg.register(_EchoTool())
    with pytest.raises(ValueError):
        reg.register(_EchoTool())


def test_get_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(ToolNotFound):
        reg.get("does_not_exist")


def test_invoke_success_persists_execution_log(ctx):
    reg = ToolRegistry()
    reg.register(_EchoTool())
    result = reg.invoke("test_echo", {"text": "hi"}, ctx)
    assert result.success
    assert result.data["text"] == "hi"

    rows = ctx.db.query("SELECT * FROM execution_log WHERE tool_name = 'test_echo'")
    assert len(rows) == 1
    assert rows[0]["success"] == 1


def test_invoke_invalid_input_fails_without_executing(ctx):
    reg = ToolRegistry()
    reg.register(_EchoTool())
    result = reg.invoke("test_echo", {}, ctx)  # missing required 'text'
    assert not result.success
    assert result.error.code == "invalid_input"


def test_invoke_catches_exceptions(ctx):
    reg = ToolRegistry()
    reg.register(_CrashingTool())
    result = reg.invoke("test_crash", {}, ctx)
    assert not result.success
    assert result.error.code == "execution_error"


def test_invoke_validates_output(ctx):
    reg = ToolRegistry()
    reg.register(_BadOutputTool())
    result = reg.invoke("test_bad_output", {}, ctx)
    assert not result.success
    assert result.error.code == "invalid_output"


def test_review_level_denied_by_default_gate(ctx):
    class ReviewTool(_EchoTool):
        name = "test_review"
        permission = PermissionLevel.REVIEW

    reg = ToolRegistry(policy=PermissionPolicy(), gate=DenyAllGate())
    reg.register(ReviewTool())
    result = reg.invoke("test_review", {"text": "hi"}, ctx)
    assert not result.success
    assert result.error.code == "approval_denied"


def test_describe_returns_json_schema_shape():
    reg = ToolRegistry()
    reg.register(_EchoTool())
    described = reg.describe()
    assert described[0]["name"] == "test_echo"
    assert described[0]["input_schema"]["type"] == "object"
