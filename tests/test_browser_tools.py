"""Tool-layer tests via `FakeBrowserAgent` — no real browser required.

Covers schema/permission wiring, argument passing, and error translation,
independent of whether Playwright/Chromium is actually installed.
`tests/test_browser_playwright.py` covers the real backend.
"""
from __future__ import annotations

import base64

from core.errors import BrowserActionFailed, BrowserUnavailable
from core.permissions.levels import PermissionLevel
from tools.browser.base import PageObservation
from tools.browser.fake import FakeBrowserAgent
from tools.browser.tools import (
    BrowserClickTool, BrowserFillTool, BrowserGetTextTool, BrowserGoBackTool,
    BrowserNavigateTool, BrowserScreenshotTool,
)


def test_navigate_returns_observation(ctx):
    fake = FakeBrowserAgent()
    result = BrowserNavigateTool(agent=fake).execute({"url": "https://example.com"}, ctx)
    assert result.success
    assert result.data["url"] == "https://example.com"
    assert fake.calls == [("navigate", ("https://example.com",))]


def test_get_text_returns_current_page_text(ctx):
    fake = FakeBrowserAgent(initial=PageObservation(url="about:blank", title="", text_excerpt="hello"))
    result = BrowserGetTextTool(agent=fake).execute({}, ctx)
    assert result.success
    assert result.data["text"] == "hello"
    assert fake.calls == [("get_text", ())]


def test_screenshot_returns_base64_png(ctx):
    fake = FakeBrowserAgent()
    result = BrowserScreenshotTool(agent=fake).execute({}, ctx)
    assert result.success
    assert base64.standard_b64decode(result.data["image_base64"]) == b"\x89PNG fake screenshot"
    assert result.data["mime_type"] == "image/png"


def test_click_calls_agent_with_selector(ctx):
    fake = FakeBrowserAgent()
    result = BrowserClickTool(agent=fake).execute({"selector": "#submit"}, ctx)
    assert result.success
    assert fake.calls == [("click", ("#submit",))]


def test_click_returns_resulting_observation(ctx):
    fake = FakeBrowserAgent()
    fake.set_observation(PageObservation(url="https://example.com/next", title="Next", text_excerpt="ok"))
    result = BrowserClickTool(agent=fake).execute({"selector": "#link"}, ctx)
    assert result.data == {"url": "https://example.com/next", "title": "Next", "text_excerpt": "ok"}


def test_click_on_missing_selector_becomes_failed_tool_result(ctx):
    fake = FakeBrowserAgent(fail_selectors={"#missing"})
    result = BrowserClickTool(agent=fake).execute({"selector": "#missing"}, ctx)
    assert not result.success
    assert result.error.code == "click_failed"


def test_fill_calls_agent_with_selector_and_text(ctx):
    fake = FakeBrowserAgent()
    result = BrowserFillTool(agent=fake).execute({"selector": "#box", "text": "hello"}, ctx)
    assert result.success
    assert fake.calls == [("fill", ("#box", "hello"))]


def test_fill_on_missing_selector_becomes_failed_tool_result(ctx):
    fake = FakeBrowserAgent(fail_selectors={"#missing"})
    result = BrowserFillTool(agent=fake).execute({"selector": "#missing", "text": "x"}, ctx)
    assert not result.success
    assert result.error.code == "fill_failed"


def test_go_back_returns_observation(ctx):
    fake = FakeBrowserAgent()
    result = BrowserGoBackTool(agent=fake).execute({}, ctx)
    assert result.success
    assert fake.calls == [("go_back", ())]


def test_browser_unavailable_becomes_failed_tool_result(ctx):
    class _UnavailableAgent(FakeBrowserAgent):
        def navigate(self, url):
            raise BrowserUnavailable("playwright is not installed")

    result = BrowserNavigateTool(agent=_UnavailableAgent()).execute({"url": "https://example.com"}, ctx)
    assert not result.success
    assert result.error.code == "browser_unavailable"


def test_screenshot_browser_unavailable_becomes_failed_tool_result(ctx):
    class _UnavailableAgent(FakeBrowserAgent):
        def screenshot(self):
            raise BrowserUnavailable("no browser binary")

    result = BrowserScreenshotTool(agent=_UnavailableAgent()).execute({}, ctx)
    assert not result.success
    assert result.error.code == "browser_unavailable"


def test_get_text_action_failed_becomes_failed_tool_result(ctx):
    class _FailingAgent(FakeBrowserAgent):
        def get_text(self):
            raise BrowserActionFailed("page crashed")

    result = BrowserGetTextTool(agent=_FailingAgent()).execute({}, ctx)
    assert not result.success
    assert result.error.code == "read_failed"


def test_go_back_action_failed_becomes_failed_tool_result(ctx):
    class _FailingAgent(FakeBrowserAgent):
        def go_back(self):
            raise BrowserActionFailed("no previous page")

    result = BrowserGoBackTool(agent=_FailingAgent()).execute({}, ctx)
    assert not result.success
    assert result.error.code == "go_back_failed"


def test_permission_levels():
    assert BrowserNavigateTool().permission == PermissionLevel.LOW
    assert BrowserGetTextTool().permission == PermissionLevel.LOW
    assert BrowserScreenshotTool().permission == PermissionLevel.LOW
    assert BrowserGoBackTool().permission == PermissionLevel.LOW

    assert BrowserClickTool().permission == PermissionLevel.REVIEW
    assert BrowserFillTool().permission == PermissionLevel.REVIEW


def test_falls_back_to_get_browser_agent_when_none_injected(ctx, monkeypatch):
    """Without an injected agent, the tool asks tools.browser.get_browser_agent()."""
    fake = FakeBrowserAgent()
    monkeypatch.setattr("tools.browser.tools.get_browser_agent", lambda: fake)
    result = BrowserGetTextTool().execute({}, ctx)
    assert result.success
    assert fake.calls[0][0] == "get_text"
