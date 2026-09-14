"""Tests against the real `FedoraAgent` — skipped unless a live X11 session
with xdotool/scrot/xclip is actually available (set DISPLAY and install
them; see docs/DEVICES.md). This is what genuinely exercises the
subprocess/X11 integration; `tests/test_computer_tools.py` covers the
tool layer with `FakeComputerAgent` instead, so that coverage doesn't
depend on having a display.
"""
from __future__ import annotations

import subprocess
import time

import pytest

from core.errors import CapabilityUnavailable
from tools.computer.base import Point
from tools.computer.fedora import FedoraAgent, is_available

pytestmark = pytest.mark.skipif(
    not is_available(), reason="no live X11 session with xdotool/scrot/xclip (see docs/DEVICES.md)"
)


@pytest.fixture
def agent() -> FedoraAgent:
    return FedoraAgent()


def test_screenshot_returns_valid_png(agent):
    data = agent.screenshot()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 100


def test_move_mouse_does_not_raise(agent):
    agent.move_mouse(Point(50, 50))


def test_clipboard_round_trip(agent):
    agent.set_clipboard("kanna test clipboard value")
    assert agent.get_clipboard() == "kanna test clipboard value"


def test_inspect_screen_reports_dimensions(agent):
    info = agent.inspect_screen()
    assert info["width"] > 0
    assert info["height"] > 0


def test_scroll_does_not_raise(agent):
    agent.scroll(0, 2)
    agent.scroll(2, 0)
    agent.scroll(0, 0)


def test_open_and_close_application(agent):
    agent.open_application("xclock")
    time.sleep(1.0)
    found = subprocess.run(["xdotool", "search", "--class", "xclock"],
                            env=agent._env(), capture_output=True)
    assert found.stdout.strip(), "xclock window did not appear after open_application"

    agent.close_application("xclock")
    time.sleep(0.5)
    gone = subprocess.run(["xdotool", "search", "--class", "xclock"],
                           env=agent._env(), capture_output=True)
    assert not gone.stdout.strip(), "xclock window still present after close_application"


def test_click_and_type_reach_the_focused_window(agent, tmp_path):
    """End-to-end: click focuses a terminal, typed text + Ctrl-D reaches the
    process reading it — the strongest real evidence click/type/key_press work."""
    out_file = tmp_path / "typed.txt"
    proc = subprocess.Popen(
        ["xterm", "-geometry", "40x10+50+50", "-e", f"bash -c 'cat > {out_file}'"],
        env=agent._env(),
    )
    try:
        time.sleep(1.5)
        agent.click(Point(70, 70))
        time.sleep(0.3)
        agent.type_text("hello from the test suite")
        agent.key_press("ctrl+d")
        time.sleep(0.5)
        assert out_file.read_text().strip() == "hello from the test suite"
    finally:
        # xterm can take a while to notice its child shell exited and tear
        # itself down after Ctrl-D; that teardown timing isn't what this
        # test is checking, so don't fail the test over slow cleanup.
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_close_application_raises_when_nothing_matches(agent):
    with pytest.raises(CapabilityUnavailable):
        agent.close_application("definitely-not-a-real-app-xyz")


def test_open_application_raises_for_unknown_executable(agent):
    with pytest.raises(CapabilityUnavailable):
        agent.open_application("definitely-not-a-real-executable-xyz")


def test_click_rejects_unknown_button(agent):
    with pytest.raises(ValueError):
        agent.click(Point(10, 10), button="not-a-button")


def test_no_display_raises_capability_unavailable(monkeypatch):
    # display=None falls back to the ambient DISPLAY env var by design (so a
    # caller can rely on the environment); force that fallback to also be
    # absent, rather than relying on this test process happening to have
    # none set.
    monkeypatch.delenv("DISPLAY", raising=False)
    agent_without_display = FedoraAgent(display=None)
    with pytest.raises(CapabilityUnavailable, match="no DISPLAY"):
        agent_without_display.screenshot()


def test_missing_binary_raises_capability_unavailable(agent, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: None)
    with pytest.raises(CapabilityUnavailable, match="not installed"):
        agent.screenshot()
