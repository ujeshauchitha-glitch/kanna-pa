"""The device-independent computer-control interface.

`ComputerAgent` is the capability surface every concrete backend (a
future `FedoraAgent`, `WindowsAgent`, `PhoneAgent`) implements. Kanna's
core never talks to a specific OS — it talks to this Protocol, and a
device is selected at the edges (see `tools/computer/null.py` for the
Phase 1 stand-in and `docs/DEVICES.md` for how device selection will
work once real backends exist).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Point:
    x: int
    y: int


class ComputerAgent(Protocol):
    """Capabilities a device backend must provide.

    Every method returns a `ToolResult`-shaped outcome via the tool that
    wraps it (see `NullComputerAgent` for the Phase 1 implementation,
    which always reports `CapabilityUnavailable` rather than pretending
    an action happened).
    """

    def screenshot(self) -> bytes: ...
    def move_mouse(self, point: Point) -> None: ...
    def click(self, point: Point, button: str = "left") -> None: ...
    def type_text(self, text: str) -> None: ...
    def key_press(self, key: str) -> None: ...
    def scroll(self, dx: int, dy: int) -> None: ...
    def get_clipboard(self) -> str: ...
    def set_clipboard(self, text: str) -> None: ...
    def open_application(self, name: str) -> None: ...
    def close_application(self, name: str) -> None: ...
    def inspect_screen(self) -> dict: ...
