"""NullComputerAgent: every method must honestly refuse, never pretend."""
from __future__ import annotations

import pytest

from core.errors import CapabilityUnavailable
from tools.computer.base import Point
from tools.computer.null import NullComputerAgent


@pytest.fixture
def agent() -> NullComputerAgent:
    return NullComputerAgent()


def test_every_method_raises_capability_unavailable(agent):
    with pytest.raises(CapabilityUnavailable):
        agent.screenshot()
    with pytest.raises(CapabilityUnavailable):
        agent.move_mouse(Point(0, 0))
    with pytest.raises(CapabilityUnavailable):
        agent.click(Point(0, 0))
    with pytest.raises(CapabilityUnavailable):
        agent.type_text("x")
    with pytest.raises(CapabilityUnavailable):
        agent.key_press("Return")
    with pytest.raises(CapabilityUnavailable):
        agent.scroll(0, 1)
    with pytest.raises(CapabilityUnavailable):
        agent.get_clipboard()
    with pytest.raises(CapabilityUnavailable):
        agent.set_clipboard("x")
    with pytest.raises(CapabilityUnavailable):
        agent.open_application("x")
    with pytest.raises(CapabilityUnavailable):
        agent.close_application("x")
    with pytest.raises(CapabilityUnavailable):
        agent.inspect_screen()


def test_error_message_explains_why(agent):
    with pytest.raises(CapabilityUnavailable, match="no device backend"):
        agent.screenshot()
