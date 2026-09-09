"""Tool-layer tests via `FakeComputerAgent` — no display required.

Covers schema/permission wiring and error translation, independent of
whether a real X11 session is available. `tests/test_fedora_agent.py`
covers the real subprocess/X11 integration.
"""
from __future__ import annotations

import base64

from core.permissions.levels import PermissionLevel
from tools.computer.base import Point
from tools.computer.fake import FakeComputerAgent
from tools.computer.tools import (
    ComputerClickTool, ComputerCloseApplicationTool, ComputerGetClipboardTool,
    ComputerInspectScreenTool, ComputerKeyPressTool, ComputerMoveMouseTool,
    ComputerOpenApplicationTool, ComputerScreenshotTool, ComputerScrollTool,
    ComputerSetClipboardTool, ComputerTypeTextTool,
)


def test_screenshot_returns_base64(ctx):
    fake = FakeComputerAgent(screenshot_bytes=b"\x89PNG fake")
    result = ComputerScreenshotTool(agent=fake).execute({}, ctx)
    assert result.success
    assert base64.standard_b64decode(result.data["image_base64"]) == b"\x89PNG fake"
    assert result.data["mime_type"] == "image/png"


def test_move_mouse_calls_agent_with_point(ctx):
    fake = FakeComputerAgent()
    result = ComputerMoveMouseTool(agent=fake).execute({"x": 10, "y": 20}, ctx)
    assert result.success
    assert fake.calls == [("move_mouse", (Point(10, 20),), {})]


def test_click_calls_agent_with_point_and_button(ctx):
    fake = FakeComputerAgent()
    result = ComputerClickTool(agent=fake).execute({"x": 5, "y": 6, "button": "right"}, ctx)
    assert result.success
    assert fake.calls == [("click", (Point(5, 6),), {"button": "right"})]


def test_click_default_button_is_left(ctx):
    fake = FakeComputerAgent()
    ComputerClickTool(agent=fake).execute({"x": 1, "y": 1}, ctx)
    assert fake.calls[0][2]["button"] == "left"


def test_type_text(ctx):
    fake = FakeComputerAgent()
    result = ComputerTypeTextTool(agent=fake).execute({"text": "hello"}, ctx)
    assert result.success
    assert fake.calls == [("type_text", ("hello",), {})]


def test_key_press(ctx):
    fake = FakeComputerAgent()
    result = ComputerKeyPressTool(agent=fake).execute({"key": "Return"}, ctx)
    assert result.success
    assert fake.calls == [("key_press", ("Return",), {})]


def test_scroll_defaults_to_zero(ctx):
    fake = FakeComputerAgent()
    result = ComputerScrollTool(agent=fake).execute({}, ctx)
    assert result.success
    assert fake.calls == [("scroll", (0, 0), {})]


def test_get_clipboard(ctx):
    fake = FakeComputerAgent(clipboard="existing text")
    result = ComputerGetClipboardTool(agent=fake).execute({}, ctx)
    assert result.success
    assert result.data["text"] == "existing text"


def test_set_clipboard(ctx):
    fake = FakeComputerAgent()
    result = ComputerSetClipboardTool(agent=fake).execute({"text": "new text"}, ctx)
    assert result.success
    assert fake.get_clipboard() == "new text"


def test_open_application(ctx):
    fake = FakeComputerAgent()
    result = ComputerOpenApplicationTool(agent=fake).execute({"name": "firefox"}, ctx)
    assert result.success
    assert fake.calls == [("open_application", ("firefox",), {})]


def test_close_application(ctx):
    fake = FakeComputerAgent()
    result = ComputerCloseApplicationTool(agent=fake).execute({"name": "firefox"}, ctx)
    assert result.success
    assert fake.calls == [("close_application", ("firefox",), {})]


def test_inspect_screen(ctx):
    fake = FakeComputerAgent(inspect_result={"width": 1920, "height": 1080, "active_window": "Terminal"})
    result = ComputerInspectScreenTool(agent=fake).execute({}, ctx)
    assert result.success
    assert result.data == {"width": 1920, "height": 1080, "active_window": "Terminal"}


def test_inspect_screen_null_active_window_becomes_empty_string(ctx):
    fake = FakeComputerAgent(inspect_result={"width": 100, "height": 100, "active_window": None})
    result = ComputerInspectScreenTool(agent=fake).execute({}, ctx)
    assert result.data["active_window"] == ""


def test_capability_unavailable_becomes_failed_tool_result(ctx):
    fake = FakeComputerAgent(raise_on={"screenshot"})
    result = ComputerScreenshotTool(agent=fake).execute({}, ctx)
    assert not result.success
    assert result.error.code == "computer_control_unavailable"


def test_click_invalid_button_reported_as_tool_failure(ctx):
    class _RaisingAgent(FakeComputerAgent):
        def click(self, point, button="left"):
            raise ValueError(f"unknown mouse button: {button!r}")

    result = ComputerClickTool(agent=_RaisingAgent()).execute({"x": 1, "y": 1, "button": "left"}, ctx)
    assert not result.success
    assert result.error.code == "invalid_button"


def test_permission_levels():
    assert ComputerScreenshotTool().permission == PermissionLevel.LOW
    assert ComputerMoveMouseTool().permission == PermissionLevel.LOW
    assert ComputerScrollTool().permission == PermissionLevel.LOW
    assert ComputerGetClipboardTool().permission == PermissionLevel.LOW
    assert ComputerSetClipboardTool().permission == PermissionLevel.LOW
    assert ComputerOpenApplicationTool().permission == PermissionLevel.LOW
    assert ComputerInspectScreenTool().permission == PermissionLevel.LOW

    assert ComputerClickTool().permission == PermissionLevel.REVIEW
    assert ComputerTypeTextTool().permission == PermissionLevel.REVIEW
    assert ComputerKeyPressTool().permission == PermissionLevel.REVIEW
    assert ComputerCloseApplicationTool().permission == PermissionLevel.REVIEW


def test_falls_back_to_get_computer_agent_when_none_injected(ctx, monkeypatch):
    """Without an injected agent, the tool asks tools.computer.get_computer_agent()."""
    fake = FakeComputerAgent()
    monkeypatch.setattr("tools.computer.tools.get_computer_agent", lambda: fake)
    result = ComputerInspectScreenTool().execute({}, ctx)
    assert result.success
    assert fake.calls[0][0] == "inspect_screen"
