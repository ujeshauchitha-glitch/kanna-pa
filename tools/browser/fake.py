"""A scripted `BrowserAgent` for tool-layer tests — no real browser required."""
from __future__ import annotations

from core.errors import BrowserActionFailed
from tools.browser.base import PageObservation


class FakeBrowserAgent:
    def __init__(self, *, initial: PageObservation | None = None,
                 fail_selectors: set[str] | None = None) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self._observation = initial or PageObservation(url="about:blank", title="", text_excerpt="")
        self._fail_selectors = fail_selectors or set()
        self.closed = False

    def _record(self, name: str, *args) -> None:
        self.calls.append((name, args))

    def navigate(self, url: str) -> PageObservation:
        self._record("navigate", url)
        self._observation = PageObservation(url=url, title=self._observation.title,
                                              text_excerpt=self._observation.text_excerpt)
        return self._observation

    def get_text(self) -> str:
        self._record("get_text")
        return self._observation.text_excerpt

    def screenshot(self) -> bytes:
        self._record("screenshot")
        return b"\x89PNG fake screenshot"

    def click(self, selector: str) -> PageObservation:
        self._record("click", selector)
        if selector in self._fail_selectors:
            raise BrowserActionFailed(f"could not click {selector!r}: no matching element")
        return self._observation

    def fill(self, selector: str, text: str) -> PageObservation:
        self._record("fill", selector, text)
        if selector in self._fail_selectors:
            raise BrowserActionFailed(f"could not fill {selector!r}: no matching element")
        return self._observation

    def go_back(self) -> PageObservation:
        self._record("go_back")
        return self._observation

    def current_url(self) -> str:
        return self._observation.url

    def close(self) -> None:
        self.closed = True

    def set_observation(self, observation: PageObservation) -> None:
        """Test helper: script what the next observation-returning call sees."""
        self._observation = observation
