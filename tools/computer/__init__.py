"""Computer control: device-independent interface + backend selection.

`get_computer_agent()` is the one place that decides which
`ComputerAgent` backend is actually in use — `FedoraAgent` when a live
X11 session and its required binaries are detected, `NullComputerAgent`
otherwise. Nothing else in Kanna hardcodes a backend; see
`docs/DEVICES.md` for the fuller device-selection story this is a first
step toward (today: one backend, real detection; not yet: routing across
multiple registered devices).
"""
from __future__ import annotations

from tools.computer.base import ComputerAgent
from tools.computer.fedora import FedoraAgent, is_available
from tools.computer.null import NullComputerAgent


def get_computer_agent() -> ComputerAgent:
    if is_available():
        return FedoraAgent()
    return NullComputerAgent()
