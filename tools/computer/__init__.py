"""Computer control: device-independent interface + backend selection.

`get_computer_agent()` is the one place that decides which
`ComputerAgent` backend is actually in use — `FedoraAgent` when a live
X11 session and its required binaries are detected, `WindowsAgent` when
running on Windows with PowerShell available, `NullComputerAgent`
otherwise. Nothing else in Kanna hardcodes a backend; see `docs/DEVICES.md`
for the fuller device-selection story this is a first step toward.
"""
from __future__ import annotations

from tools.computer.base import ComputerAgent
from tools.computer.fedora import FedoraAgent
from tools.computer.fedora import is_available as fedora_available
from tools.computer.null import NullComputerAgent
from tools.computer.windows import WindowsAgent
from tools.computer.windows import is_available as windows_available


def get_computer_agent() -> ComputerAgent:
    if fedora_available():
        return FedoraAgent()
    if windows_available():
        return WindowsAgent()
    return NullComputerAgent()
