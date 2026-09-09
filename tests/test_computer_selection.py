"""`tools.computer.get_computer_agent()` backend selection."""
from __future__ import annotations

from tools.computer import get_computer_agent
from tools.computer.fedora import FedoraAgent
from tools.computer.null import NullComputerAgent


def test_selects_fedora_agent_when_available(monkeypatch):
    monkeypatch.setattr("tools.computer.is_available", lambda: True)
    assert isinstance(get_computer_agent(), FedoraAgent)


def test_selects_null_agent_when_unavailable(monkeypatch):
    monkeypatch.setattr("tools.computer.is_available", lambda: False)
    assert isinstance(get_computer_agent(), NullComputerAgent)
