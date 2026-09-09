"""Browser automation capabilities exposed through the tool registry.

`navigate`/`click`/`fill`/`go_back` all return the resulting
`PageObservation` in `ToolResult.data` — never just "ok" — so a plan
step, or an LLM reading the result, has real evidence of what happened
rather than an assumption that the action worked. Permission split
follows the same principle as `tools/computer/tools.py`: reading is LOW,
actions whose consequence Kanna can't know (click, fill — could submit
a form, trigger a purchase, send something) are REVIEW.
"""
from __future__ import annotations

import base64

from core.errors import BrowserActionFailed, BrowserUnavailable
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.registry import ToolRegistry
from core.tools.result import ToolResult
from core.tools.schema import obj, string
from tools.browser import get_browser_agent
from tools.browser.base import BrowserAgent, PageObservation

_OBSERVATION_SCHEMA = obj({"url": string(), "title": string(), "text_excerpt": string()})


def _observation_data(obs: PageObservation) -> dict:
    return {"url": obs.url, "title": obs.title, "text_excerpt": obs.text_excerpt}


class _BrowserTool:
    def __init__(self, agent: BrowserAgent | None = None) -> None:
        self._agent_override = agent

    def _agent(self, ctx: ToolContext) -> BrowserAgent:
        return self._agent_override or get_browser_agent()

    @staticmethod
    def _fail(exc: BrowserUnavailable | BrowserActionFailed, code: str) -> ToolResult:
        return ToolResult.fail(code, str(exc))


class BrowserNavigateTool(_BrowserTool):
    name = "browser_navigate"
    description = "Navigate the browser to a URL and observe the resulting page."
    permission = PermissionLevel.LOW
    input_schema = obj({"url": string()}, required=("url",))
    output_schema = _OBSERVATION_SCHEMA

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            obs = self._agent(ctx).navigate(args["url"])
        except BrowserUnavailable as exc:
            return self._fail(exc, "browser_unavailable")
        except BrowserActionFailed as exc:
            return self._fail(exc, "navigation_failed")
        return ToolResult.ok(_observation_data(obs))


class BrowserGetTextTool(_BrowserTool):
    name = "browser_get_text"
    description = "Read the current page's visible text content."
    permission = PermissionLevel.LOW
    input_schema = obj({})
    output_schema = obj({"text": string()})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            text = self._agent(ctx).get_text()
        except BrowserUnavailable as exc:
            return self._fail(exc, "browser_unavailable")
        except BrowserActionFailed as exc:
            return self._fail(exc, "read_failed")
        return ToolResult.ok({"text": text})


class BrowserScreenshotTool(_BrowserTool):
    name = "browser_screenshot"
    description = "Capture a screenshot of the current page."
    permission = PermissionLevel.LOW
    input_schema = obj({})
    output_schema = obj({"image_base64": string(), "mime_type": string()})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            image_bytes = self._agent(ctx).screenshot()
        except BrowserUnavailable as exc:
            return self._fail(exc, "browser_unavailable")
        except BrowserActionFailed as exc:
            return self._fail(exc, "screenshot_failed")
        return ToolResult.ok({
            "image_base64": base64.standard_b64encode(image_bytes).decode("ascii"),
            "mime_type": "image/png",
        })


class BrowserClickTool(_BrowserTool):
    name = "browser_click"
    description = ("Click an element matching a CSS selector, then observe the resulting page. "
                    "Requires approval — the consequence of a click is unknown to Kanna.")
    permission = PermissionLevel.REVIEW
    input_schema = obj({"selector": string(description="CSS selector")}, required=("selector",))
    output_schema = _OBSERVATION_SCHEMA

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            obs = self._agent(ctx).click(args["selector"])
        except BrowserUnavailable as exc:
            return self._fail(exc, "browser_unavailable")
        except BrowserActionFailed as exc:
            return self._fail(exc, "click_failed")
        return ToolResult.ok(_observation_data(obs))


class BrowserFillTool(_BrowserTool):
    name = "browser_fill"
    description = ("Fill a form field matching a CSS selector, then observe the resulting page. "
                    "Requires approval — the destination/consequence is unknown to Kanna.")
    permission = PermissionLevel.REVIEW
    input_schema = obj(
        {"selector": string(description="CSS selector"), "text": string()},
        required=("selector", "text"),
    )
    output_schema = _OBSERVATION_SCHEMA

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            obs = self._agent(ctx).fill(args["selector"], args["text"])
        except BrowserUnavailable as exc:
            return self._fail(exc, "browser_unavailable")
        except BrowserActionFailed as exc:
            return self._fail(exc, "fill_failed")
        return ToolResult.ok(_observation_data(obs))


class BrowserGoBackTool(_BrowserTool):
    name = "browser_go_back"
    description = "Navigate back to the previous page and observe the result."
    permission = PermissionLevel.LOW
    input_schema = obj({})
    output_schema = _OBSERVATION_SCHEMA

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            obs = self._agent(ctx).go_back()
        except BrowserUnavailable as exc:
            return self._fail(exc, "browser_unavailable")
        except BrowserActionFailed as exc:
            return self._fail(exc, "go_back_failed")
        return ToolResult.ok(_observation_data(obs))


ALL_TOOLS = [
    BrowserNavigateTool(), BrowserGetTextTool(), BrowserScreenshotTool(), BrowserClickTool(),
    BrowserFillTool(), BrowserGoBackTool(),
]


def register_all(registry: ToolRegistry) -> None:
    for tool in ALL_TOOLS:
        registry.register(tool)
