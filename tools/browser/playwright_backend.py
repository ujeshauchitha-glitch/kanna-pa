"""Real browser automation, backed by Playwright + Chromium.

`playwright` is lazily imported (extra `[browser]`) so nothing outside
this module needs it installed — same discipline as
`core/llm/anthropic_provider.py` and `vision/_common.py`. Raises
`BrowserUnavailable` (package/browser missing) or `BrowserActionFailed`
(a specific navigate/click/fill didn't work) rather than crashing or
silently no-op'ing — see `tools/browser/base.py` for why every
state-changing method returns a `PageObservation`.

A single Chromium page is kept alive across calls (`_ensure_page()`
launches once, reuses after) — that's what makes "navigate, then click,
then read the result" behave like an actual browsing session instead of
a fresh tab every time. `close()` tears it down; `tools/browser/
__init__.py` owns the one long-lived instance a process uses.
"""
from __future__ import annotations

import os
from pathlib import Path

from core.errors import BrowserActionFailed, BrowserUnavailable
from tools.browser.base import PageObservation

_MAX_TEXT_EXCERPT = 4000


def _find_chromium_executable() -> str | None:
    """Resolve a pre-staged Chromium binary if the environment provides one.

    Some sandboxes pin `PLAYWRIGHT_BROWSERS_PATH` to a Chromium build
    that predates whatever `playwright` version is pip-installed, with a
    `chromium` convenience symlink at its root for exactly this case
    (see the environment notes this project was built in). A normal
    install (where `playwright install` matched the installed package)
    doesn't need this — Playwright resolves its own browser by default —
    so this only overrides when that symlink is actually there.
    """
    browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if not browsers_path:
        return None
    candidate = Path(browsers_path) / "chromium"
    return str(candidate) if candidate.exists() else None


class PlaywrightBrowserAgent:
    def __init__(self, *, headless: bool = True, timeout_ms: int = 30_000) -> None:
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._playwright = None
        self._browser = None
        self._page = None

    def _ensure_page(self):
        if self._page is not None:
            return self._page

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserUnavailable(
                "the 'playwright' package is not installed; run `pip install kanna[browser]` "
                "then `playwright install chromium`"
            ) from exc

        try:
            self._playwright = sync_playwright().start()
            launch_kwargs = {
                "headless": self.headless,
                "args": ["--disable-background-networking", "--no-first-run"],
            }
            executable = _find_chromium_executable()
            if executable:
                launch_kwargs["executable_path"] = executable
            self._browser = self._playwright.chromium.launch(**launch_kwargs)
            self._page = self._browser.new_page()
            self._page.set_default_timeout(self.timeout_ms)
        except Exception as exc:  # noqa: BLE001 - any launch failure means "unavailable"
            self.close()
            raise BrowserUnavailable(f"could not launch a browser: {exc}") from exc

        return self._page

    def _observe(self) -> PageObservation:
        page = self._page
        try:
            text = page.inner_text("body")
        except Exception:  # noqa: BLE001 - a page with no <body> yet (about:blank) shouldn't crash this
            text = ""
        return PageObservation(url=page.url, title=page.title(), text_excerpt=text[:_MAX_TEXT_EXCERPT])

    def navigate(self, url: str) -> PageObservation:
        page = self._ensure_page()
        try:
            page.goto(url)
        except Exception as exc:  # noqa: BLE001 - playwright.Error, TimeoutError, etc.
            raise BrowserActionFailed(f"could not navigate to {url!r}: {exc}") from exc
        return self._observe()

    def get_text(self) -> str:
        page = self._ensure_page()
        try:
            return page.inner_text("body")
        except Exception as exc:  # noqa: BLE001
            raise BrowserActionFailed(f"could not read page text: {exc}") from exc

    def screenshot(self) -> bytes:
        page = self._ensure_page()
        try:
            return page.screenshot()
        except Exception as exc:  # noqa: BLE001
            raise BrowserActionFailed(f"could not take a screenshot: {exc}") from exc

    def click(self, selector: str) -> PageObservation:
        page = self._ensure_page()
        try:
            page.click(selector)
        except Exception as exc:  # noqa: BLE001
            raise BrowserActionFailed(f"could not click {selector!r}: {exc}") from exc
        return self._observe()

    def fill(self, selector: str, text: str) -> PageObservation:
        page = self._ensure_page()
        try:
            page.fill(selector, text)
        except Exception as exc:  # noqa: BLE001
            raise BrowserActionFailed(f"could not fill {selector!r}: {exc}") from exc
        return self._observe()

    def go_back(self) -> PageObservation:
        page = self._ensure_page()
        try:
            page.go_back()
        except Exception as exc:  # noqa: BLE001
            raise BrowserActionFailed(f"could not go back: {exc}") from exc
        return self._observe()

    def current_url(self) -> str:
        return self._ensure_page().url

    def close(self) -> None:
        for obj, method in ((self._browser, "close"), (self._playwright, "stop")):
            if obj is not None:
                try:
                    getattr(obj, method)()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
        self._page = None
        self._browser = None
        self._playwright = None
