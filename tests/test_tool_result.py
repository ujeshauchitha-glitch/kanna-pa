from __future__ import annotations

from core.tools.result import ToolResult


def test_ok_result():
    result = ToolResult.ok({"a": 1})
    assert result.success
    assert result.data == {"a": 1}
    assert result.error is None


def test_fail_result():
    result = ToolResult.fail("bad_input", "nope", details={"field": "x"})
    assert not result.success
    assert result.error.code == "bad_input"
    assert result.error.details == {"field": "x"}


def test_to_dict_roundtrip_shape():
    result = ToolResult.ok({"a": 1}, files_created=["a.txt"])
    d = result.to_dict()
    assert d["success"] is True
    assert d["files_created"] == ["a.txt"]
    assert d["error"] is None
