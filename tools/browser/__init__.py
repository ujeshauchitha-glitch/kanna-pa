"""Browser automation: interface + the one long-lived session a process uses.

Unlike `tools.computer.get_computer_agent()` (stateless — a fresh check
every call is fine), a browser needs to stay the *same* open page across
calls for "navigate, then click, then read the result" to mean anything.
`get_browser_agent()` is a process-level singleton for that reason;
`reset_browser_agent()` tears it down (tests use this to start clean).
"""
from __future__ import annotations

import importlib.util

from tools.browser.base import BrowserAgent
from tools.browser.playwright_backend import PlaywrightBrowserAgent

_agent: BrowserAgent | None = None


def is_available() -> bool:
    """True if the `playwright` package is importable.

    Doesn't guarantee a browser binary is actually installed — that's
    only knowable by trying to launch one, which `get_browser_agent()`'s
    first real call does, raising `BrowserUnavailable` if it fails. This
    check exists so callers can short-circuit the obvious "package not
    installed at all" case without paying for a launch attempt.
    """
    return importlib.util.find_spec("playwright") is not None


def get_browser_agent() -> BrowserAgent:
    global _agent
    if _agent is None:
        _agent = PlaywrightBrowserAgent()
    return _agent


def reset_browser_agent() -> None:
    """Close and discard the current session, if any. Tests use this between cases."""
    global _agent
    if _agent is not None:
        _agent.close()
    _agent = None
