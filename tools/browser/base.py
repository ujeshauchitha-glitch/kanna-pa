"""The browser automation interface.

"Browser automation must support observation after actions. Never assume
a click succeeded." — every state-changing method (`navigate`, `click`,
`fill`, `go_back`) returns a `PageObservation` of what the page actually
looks like *after* the action, rather than a bare `None` the caller has
to trust blindly. A tool wrapping these (`tools/browser/tools.py`)
surfaces that observation in its `ToolResult.data`, so a plan step's
`expected` postcondition — or an LLM planner reading the result — has
something real to check against, not just "the call didn't raise".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class PageObservation:
    url: str
    title: str
    text_excerpt: str  # truncated visible text — enough to see what actually changed


class BrowserAgent(Protocol):
    def navigate(self, url: str) -> PageObservation: ...
    def get_text(self) -> str: ...
    def screenshot(self) -> bytes: ...
    def click(self, selector: str) -> PageObservation: ...
    def fill(self, selector: str, text: str) -> PageObservation: ...
    def go_back(self) -> PageObservation: ...
    def current_url(self) -> str: ...
    def close(self) -> None: ...
