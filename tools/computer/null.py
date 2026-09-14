"""The Phase 1 computer-control backend: honestly reports it has none.

No `FedoraAgent`/`WindowsAgent`/`PhoneAgent` exists yet (Phase 2+). Rather
than fake a click or a screenshot, every method here raises
`CapabilityUnavailable` — the agent loop is expected to catch this and
report the real blocker instead of claiming success.
"""
from __future__ import annotations

from core.errors import CapabilityUnavailable
from tools.computer.base import Point


class NullComputerAgent:
    """Implements `ComputerAgent` by refusing every action with a clear reason."""

    def _unavailable(self, action: str) -> CapabilityUnavailable:
        return CapabilityUnavailable(
            f"computer-control action '{action}' is unavailable: no device backend is configured "
            f"for this environment (Fedora/Windows/Phone agents are not yet implemented)"
        )

    def screenshot(self) -> bytes:
        raise self._unavailable("screenshot")

    def move_mouse(self, point: Point) -> None:
        raise self._unavailable("move_mouse")

    def click(self, point: Point, button: str = "left") -> None:
        raise self._unavailable("click")

    def type_text(self, text: str) -> None:
        raise self._unavailable("type_text")

    def key_press(self, key: str) -> None:
        raise self._unavailable("key_press")

    def scroll(self, dx: int, dy: int) -> None:
        raise self._unavailable("scroll")

    def get_clipboard(self) -> str:
        raise self._unavailable("get_clipboard")

    def set_clipboard(self, text: str) -> None:
        raise self._unavailable("set_clipboard")

    def open_application(self, name: str) -> None:
        raise self._unavailable("open_application")

    def close_application(self, name: str) -> None:
        raise self._unavailable("close_application")

    def inspect_screen(self) -> dict:
        raise self._unavailable("inspect_screen")
