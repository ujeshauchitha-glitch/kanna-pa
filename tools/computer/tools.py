"""Computer-control capabilities exposed through the tool registry.

Each tool wraps one `ComputerAgent` method. Read-only/reversible actions
(screenshot, inspect, clipboard read, mouse move, scroll, launching an
app) are LOW; actions Kanna can't know the consequence of — a click or
keystroke could submit a form, send a message, or trigger any other
irreversible external action, and closing an app can lose unsaved work —
are REVIEW, per the same "irreversible or externally-visible action"
principle `docs/SECURITY.md` applies everywhere else.

Every tool resolves its `ComputerAgent` at call time via `_agent()`, so
whichever backend `core.bootstrap` wired up (currently `FedoraAgent` when
available, else `NullComputerAgent`) is used transparently — a tool never
hardcodes which backend it's talking to.
"""
from __future__ import annotations

import base64

from core.errors import CapabilityUnavailable
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.registry import ToolRegistry
from core.tools.result import ToolResult
from core.tools.schema import integer, obj, string
from tools.computer import get_computer_agent
from tools.computer.base import ComputerAgent, Point


class _ComputerTool:
    def __init__(self, agent: ComputerAgent | None = None) -> None:
        self._agent_override = agent

    def _agent(self, ctx: ToolContext) -> ComputerAgent:
        return self._agent_override or get_computer_agent()

    @staticmethod
    def _fail(exc: CapabilityUnavailable) -> ToolResult:
        return ToolResult.fail("computer_control_unavailable", str(exc))


class ComputerScreenshotTool(_ComputerTool):
    name = "computer_screenshot"
    description = "Capture a screenshot of the current display."
    permission = PermissionLevel.LOW
    input_schema = obj({})
    output_schema = obj({"image_base64": string(), "mime_type": string()})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            image_bytes = self._agent(ctx).screenshot()
        except CapabilityUnavailable as exc:
            return self._fail(exc)
        return ToolResult.ok({
            "image_base64": base64.standard_b64encode(image_bytes).decode("ascii"),
            "mime_type": "image/png",
        })


class ComputerMoveMouseTool(_ComputerTool):
    name = "computer_move_mouse"
    description = "Move the mouse cursor to (x, y)."
    permission = PermissionLevel.LOW
    input_schema = obj({"x": integer(minimum=0), "y": integer(minimum=0)}, required=("x", "y"))
    output_schema = obj({})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            self._agent(ctx).move_mouse(Point(args["x"], args["y"]))
        except CapabilityUnavailable as exc:
            return self._fail(exc)
        return ToolResult.ok({})


class ComputerClickTool(_ComputerTool):
    name = "computer_click"
    description = "Click the mouse at (x, y). Consequences are unknown to Kanna, so this requires approval."
    permission = PermissionLevel.REVIEW
    input_schema = obj(
        {"x": integer(minimum=0), "y": integer(minimum=0),
         "button": string(enum=("left", "middle", "right"), default="left")},
        required=("x", "y"),
    )
    output_schema = obj({})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            self._agent(ctx).click(Point(args["x"], args["y"]), button=args.get("button", "left"))
        except (CapabilityUnavailable, ValueError) as exc:
            if isinstance(exc, ValueError):
                return ToolResult.fail("invalid_button", str(exc))
            return self._fail(exc)
        return ToolResult.ok({})


class ComputerTypeTextTool(_ComputerTool):
    name = "computer_type_text"
    description = "Type text at the current keyboard focus. Requires approval — the destination is unknown."
    permission = PermissionLevel.REVIEW
    input_schema = obj({"text": string()}, required=("text",))
    output_schema = obj({})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            self._agent(ctx).type_text(args["text"])
        except CapabilityUnavailable as exc:
            return self._fail(exc)
        return ToolResult.ok({})


class ComputerKeyPressTool(_ComputerTool):
    name = "computer_key_press"
    description = ("Press a key or key combination (xdotool syntax, e.g. 'Return', 'ctrl+c'). "
                    "Requires approval — could submit a form or trigger a shortcut.")
    permission = PermissionLevel.REVIEW
    input_schema = obj({"key": string()}, required=("key",))
    output_schema = obj({})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            self._agent(ctx).key_press(args["key"])
        except CapabilityUnavailable as exc:
            return self._fail(exc)
        return ToolResult.ok({})


class ComputerScrollTool(_ComputerTool):
    name = "computer_scroll"
    description = "Scroll the view — positive dy scrolls down, positive dx scrolls right."
    permission = PermissionLevel.LOW
    input_schema = obj({"dx": integer(default=0), "dy": integer(default=0)})
    output_schema = obj({})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            self._agent(ctx).scroll(int(args.get("dx", 0)), int(args.get("dy", 0)))
        except CapabilityUnavailable as exc:
            return self._fail(exc)
        return ToolResult.ok({})


class ComputerGetClipboardTool(_ComputerTool):
    name = "computer_get_clipboard"
    description = "Read the current clipboard contents."
    permission = PermissionLevel.LOW
    input_schema = obj({})
    output_schema = obj({"text": string()})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            text = self._agent(ctx).get_clipboard()
        except CapabilityUnavailable as exc:
            return self._fail(exc)
        return ToolResult.ok({"text": text})


class ComputerSetClipboardTool(_ComputerTool):
    name = "computer_set_clipboard"
    description = "Set the clipboard contents."
    permission = PermissionLevel.LOW
    input_schema = obj({"text": string()}, required=("text",))
    output_schema = obj({})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            self._agent(ctx).set_clipboard(args["text"])
        except CapabilityUnavailable as exc:
            return self._fail(exc)
        return ToolResult.ok({})


class ComputerOpenApplicationTool(_ComputerTool):
    name = "computer_open_application"
    description = "Launch an application by executable name."
    permission = PermissionLevel.LOW
    input_schema = obj({"name": string()}, required=("name",))
    output_schema = obj({})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            self._agent(ctx).open_application(args["name"])
        except CapabilityUnavailable as exc:
            return self._fail(exc)
        return ToolResult.ok({})


class ComputerCloseApplicationTool(_ComputerTool):
    name = "computer_close_application"
    description = "Close an application's window(s) by name. Requires approval — may lose unsaved work."
    permission = PermissionLevel.REVIEW
    input_schema = obj({"name": string()}, required=("name",))
    output_schema = obj({})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            self._agent(ctx).close_application(args["name"])
        except CapabilityUnavailable as exc:
            return self._fail(exc)
        return ToolResult.ok({})


class ComputerInspectScreenTool(_ComputerTool):
    name = "computer_inspect_screen"
    description = "Inspect the current screen: dimensions and the active window's title."
    permission = PermissionLevel.LOW
    input_schema = obj({})
    output_schema = obj({"width": integer(), "height": integer(), "active_window": string()})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            info = self._agent(ctx).inspect_screen()
        except CapabilityUnavailable as exc:
            return self._fail(exc)
        return ToolResult.ok({
            "width": info.get("width", 0), "height": info.get("height", 0),
            "active_window": info.get("active_window") or "",
        })


ALL_TOOLS = [
    ComputerScreenshotTool(), ComputerMoveMouseTool(), ComputerClickTool(), ComputerTypeTextTool(),
    ComputerKeyPressTool(), ComputerScrollTool(), ComputerGetClipboardTool(),
    ComputerSetClipboardTool(), ComputerOpenApplicationTool(), ComputerCloseApplicationTool(),
    ComputerInspectScreenTool(),
]


def register_all(registry: ToolRegistry) -> None:
    for tool in ALL_TOOLS:
        registry.register(tool)
