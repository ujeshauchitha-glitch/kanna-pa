"""Phone-control capabilities exposed through the tool registry.

A deliberate subset of `computer_*` — `move_mouse` (a touchscreen has no
cursor) and clipboard (no reliable adb-only API on a stock device) are
never registered here at all, rather than exposing a tool that can
structurally never succeed. `phone_tap` takes `long_press` instead of
desktop's `button="left"|"right"|"middle"` — a touchscreen has no
right-click, and long-press is the actual gesture a phone user or an
app developer would recognize.

Every tool resolves its `AdbPhoneAgent` at call time via `_agent()`,
targeting whichever device `adb devices` currently reports — a plan
built before a phone was connected still does the right thing. See
`tools/computer/phone.py`'s module docstring for this backend's
validation status before trusting it the way `computer_*`'s backends
are trusted.
"""
from __future__ import annotations

import base64

from core.errors import CapabilityUnavailable
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.registry import ToolRegistry
from core.tools.result import ToolResult
from core.tools.schema import boolean, integer, obj, string
from tools.computer.base import Point
from tools.computer.phone import AdbPhoneAgent, get_phone_agent


class _PhoneTool:
    def __init__(self, agent: AdbPhoneAgent | None = None) -> None:
        self._agent_override = agent

    def _agent(self, ctx: ToolContext) -> AdbPhoneAgent:
        return self._agent_override or get_phone_agent()

    @staticmethod
    def _fail(exc: CapabilityUnavailable) -> ToolResult:
        return ToolResult.fail("phone_control_unavailable", str(exc))


class PhoneScreenshotTool(_PhoneTool):
    name = "phone_screenshot"
    description = "Capture a screenshot of the connected Android device's screen."
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


class PhoneTapTool(_PhoneTool):
    name = "phone_tap"
    description = ("Tap the screen at (x, y), or long-press if long_press is true. "
                    "Requires approval — Kanna can't know what a tap will do.")
    permission = PermissionLevel.REVIEW
    input_schema = obj(
        {"x": integer(minimum=0), "y": integer(minimum=0),
         "long_press": boolean(default=False)},
        required=("x", "y"),
    )
    output_schema = obj({})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            self._agent(ctx).click(Point(args["x"], args["y"]),
                                   button="right" if args.get("long_press") else "left")
        except CapabilityUnavailable as exc:
            return self._fail(exc)
        return ToolResult.ok({})


class PhoneTypeTextTool(_PhoneTool):
    name = "phone_type_text"
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


class PhoneKeyPressTool(_PhoneTool):
    name = "phone_key_press"
    description = ("Press a key (e.g. 'back', 'home', 'enter'). "
                    "Requires approval — could submit a form or navigate away from work in progress.")
    permission = PermissionLevel.REVIEW
    input_schema = obj({"key": string()}, required=("key",))
    output_schema = obj({})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            self._agent(ctx).key_press(args["key"])
        except CapabilityUnavailable as exc:
            return self._fail(exc)
        return ToolResult.ok({})


class PhoneScrollTool(_PhoneTool):
    name = "phone_scroll"
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


class PhoneOpenAppTool(_PhoneTool):
    name = "phone_open_app"
    description = "Launch an app by package name (e.g. com.android.chrome) or a name to search installed packages for."
    permission = PermissionLevel.LOW
    input_schema = obj({"name": string()}, required=("name",))
    output_schema = obj({})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            self._agent(ctx).open_application(args["name"])
        except CapabilityUnavailable as exc:
            return self._fail(exc)
        return ToolResult.ok({})


class PhoneCloseAppTool(_PhoneTool):
    name = "phone_close_app"
    description = "Force-stop an app by package or search name. Requires approval — may lose unsaved work."
    permission = PermissionLevel.REVIEW
    input_schema = obj({"name": string()}, required=("name",))
    output_schema = obj({})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            self._agent(ctx).close_application(args["name"])
        except CapabilityUnavailable as exc:
            return self._fail(exc)
        return ToolResult.ok({})


class PhoneInspectScreenTool(_PhoneTool):
    name = "phone_inspect_screen"
    description = "Inspect the phone screen: dimensions and the foreground app's package name."
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
    PhoneScreenshotTool(), PhoneTapTool(), PhoneTypeTextTool(), PhoneKeyPressTool(),
    PhoneScrollTool(), PhoneOpenAppTool(), PhoneCloseAppTool(), PhoneInspectScreenTool(),
]


def register_all(registry: ToolRegistry) -> None:
    for tool in ALL_TOOLS:
        registry.register(tool)
