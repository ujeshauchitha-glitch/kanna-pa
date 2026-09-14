"""A scripted `ComputerAgent` for tool-layer tests — no display required.

Records every call (so a test can assert what the tool asked the agent
to do) and returns configured/default values. `tests/test_fedora_agent.py`
is what actually exercises `FedoraAgent` against a live X11 session
(skipped when none is available); this fake exists so the *tool* layer
— schema validation, permission wiring, error translation — has
deterministic, environment-independent coverage too.
"""
from __future__ import annotations

from core.errors import CapabilityUnavailable
from tools.computer.base import Point


class FakeComputerAgent:
    def __init__(self, *, screenshot_bytes: bytes = b"fake-png-bytes", clipboard: str = "",
                 inspect_result: dict | None = None, raise_on: set[str] | None = None) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self._screenshot_bytes = screenshot_bytes
        self._clipboard = clipboard
        self._inspect_result = inspect_result or {"width": 1280, "height": 800, "active_window": None}
        self._raise_on = raise_on or set()

    def _record(self, name: str, *args, **kwargs) -> None:
        self.calls.append((name, args, kwargs))
        if name in self._raise_on:
            raise CapabilityUnavailable(f"fake failure for {name}")

    def screenshot(self) -> bytes:
        self._record("screenshot")
        return self._screenshot_bytes

    def move_mouse(self, point: Point) -> None:
        self._record("move_mouse", point)

    def click(self, point: Point, button: str = "left") -> None:
        self._record("click", point, button=button)

    def type_text(self, text: str) -> None:
        self._record("type_text", text)

    def key_press(self, key: str) -> None:
        self._record("key_press", key)

    def scroll(self, dx: int, dy: int) -> None:
        self._record("scroll", dx, dy)

    def get_clipboard(self) -> str:
        self._record("get_clipboard")
        return self._clipboard

    def set_clipboard(self, text: str) -> None:
        self._record("set_clipboard", text)
        self._clipboard = text

    def open_application(self, name: str) -> None:
        self._record("open_application", name)

    def close_application(self, name: str) -> None:
        self._record("close_application", name)

    def inspect_screen(self) -> dict:
        self._record("inspect_screen")
        return self._inspect_result
