"""`tools.computer.get_computer_agent()` backend selection."""
from __future__ import annotations

from tools.computer import get_computer_agent
from tools.computer.fedora import FedoraAgent
from tools.computer.null import NullComputerAgent
from tools.computer.windows import WindowsAgent


def test_selects_fedora_agent_when_available(monkeypatch):
    monkeypatch.setattr("tools.computer.fedora_available", lambda: True)
    monkeypatch.setattr("tools.computer.windows_available", lambda: False)
    assert isinstance(get_computer_agent(), FedoraAgent)


def test_selects_windows_agent_when_on_windows(monkeypatch):
    monkeypatch.setattr("tools.computer.fedora_available", lambda: False)
    monkeypatch.setattr("tools.computer.windows_available", lambda: True)
    assert isinstance(get_computer_agent(), WindowsAgent)


def test_prefers_fedora_over_windows(monkeypatch):
    monkeypatch.setattr("tools.computer.fedora_available", lambda: True)
    monkeypatch.setattr("tools.computer.windows_available", lambda: True)
    assert isinstance(get_computer_agent(), FedoraAgent)


def test_selects_null_agent_when_nothing_available(monkeypatch):
    monkeypatch.setattr("tools.computer.fedora_available", lambda: False)
    monkeypatch.setattr("tools.computer.windows_available", lambda: False)
    assert isinstance(get_computer_agent(), NullComputerAgent)
