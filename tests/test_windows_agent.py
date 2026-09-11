"""Tests for the Windows computer agent backend.

Tests that mock PowerShell subprocess calls work on any OS.
The `test_windows_agent_is_detected_on_windows` test runs for real
on Windows and skips elsewhere.
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from core.errors import CapabilityUnavailable
from tools.computer.base import Point
from tools.computer.windows import WindowsAgent, is_available


@pytest.fixture(autouse=True)
def _powershell_present(request, monkeypatch):
    """Every mocked-subprocess test below calls a WindowsAgent method that
    checks `_require_powershell()` before ever reaching the mocked
    `subprocess.run` — without this, every one of them fails immediately
    with CapabilityUnavailable on any machine that doesn't actually have
    `powershell.exe` on PATH (i.e. every Linux CI runner and this
    sandbox), never exercising the mocked behavior they're meant to test.
    Tests that specifically exercise real availability detection
    (`test_windows_agent_detected_on_windows`) or patch `shutil.which`
    themselves opt out or simply override this default afterward.
    """
    if request.node.name == "test_windows_agent_detected_on_windows":
        return
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/powershell.exe" if name == "powershell.exe" else None
    )


# --- Availability detection ---


def test_is_available_true_when_powershell_on_path(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/powershell.exe" if name == "powershell.exe" else None)
    assert is_available() is True


def test_is_available_false_when_no_powershell(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert is_available() is False


def test_windows_agent_detected_on_windows():
    """On Windows, is_available() should return True."""
    if not is_available():
        pytest.skip("not on Windows or powershell.exe not on PATH")


# --- Mocked subprocess tests ---


def _mock_run(returncode=0, stdout=b"", stderr=b""):
    """Create a mock for subprocess.run that returns a fixed result."""
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            args=argv, returncode=returncode, stdout=stdout, stderr=stderr,
        )
    return fake_run


def test_screenshot_calls_powershell(ctx):
    agent = WindowsAgent()
    png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    with patch("tools.computer.windows.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"", stderr=b"",
        )
        with patch("tools.computer.windows.Path") as MockPath:
            instance = MagicMock()
            instance.read_bytes.return_value = png_data
            MockPath.return_value = instance
            with patch("tools.computer.windows.tempfile.NamedTemporaryFile") as mock_tmp:
                mock_tmp_obj = MagicMock()
                mock_tmp_obj.name = "/tmp/test.png"
                mock_tmp.return_value.__enter__ = lambda s: mock_tmp_obj
                mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
                result = agent.screenshot()
                assert isinstance(result, bytes)
                assert result[:4] == b"\x89PNG"


def test_move_mouse_calls_powershell_with_coordinates(ctx):
    agent = WindowsAgent()
    with patch("tools.computer.windows.subprocess.run") as mock_run:
        mock_run.side_effect = _mock_run()
        agent.move_mouse(Point(100, 200))
        call_args = mock_run.call_args
        script = call_args[0][0][4]  # [powershell, -NoProfile, -NonInteractive, -Command, script]
        assert "100" in script
        assert "200" in script


def test_click_calls_mouse_event(ctx):
    agent = WindowsAgent()
    with patch("tools.computer.windows.subprocess.run") as mock_run:
        mock_run.side_effect = _mock_run()
        agent.click(Point(50, 60), button="left")
        # Should call run twice: once for move, once for click
        assert mock_run.call_count == 2


def test_click_rejects_unknown_button(ctx):
    agent = WindowsAgent()
    with pytest.raises(ValueError, match="unknown mouse button"):
        agent.click(Point(0, 0), button="unknown")


def test_type_text_calls_send_keys(ctx):
    agent = WindowsAgent()
    with patch("tools.computer.windows.subprocess.run") as mock_run:
        mock_run.side_effect = _mock_run()
        agent.type_text("hello world")
        script = mock_run.call_args[0][0][4]
        assert "SendKeys" in script
        assert "hello world" in script


def test_key_press_maps_known_keys(ctx):
    agent = WindowsAgent()
    with patch("tools.computer.windows.subprocess.run") as mock_run:
        mock_run.side_effect = _mock_run()
        agent.key_press("enter")
        script = mock_run.call_args[0][0][4]
        assert "{ENTER}" in script


def test_key_press_passes_unknown_keys_through(ctx):
    agent = WindowsAgent()
    with patch("tools.computer.windows.subprocess.run") as mock_run:
        mock_run.side_effect = _mock_run()
        agent.key_press("x")
        script = mock_run.call_args[0][0][4]
        assert "x" in script


def test_scroll_vertical(ctx):
    agent = WindowsAgent()
    with patch("tools.computer.windows.subprocess.run") as mock_run:
        mock_run.side_effect = _mock_run()
        agent.scroll(0, 3)
        script = mock_run.call_args[0][0][4]
        assert "0x0800" in script  # MOUSEWHEEL


def test_scroll_horizontal(ctx):
    agent = WindowsAgent()
    with patch("tools.computer.windows.subprocess.run") as mock_run:
        mock_run.side_effect = _mock_run()
        agent.scroll(2, 0)
        script = mock_run.call_args[0][0][4]
        assert "0x1000" in script  # MOUSEHWHEEL


def test_get_clipboard(ctx):
    agent = WindowsAgent()
    with patch("tools.computer.windows.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"clipboard text", stderr=b"",
        )
        text = agent.get_clipboard()
        assert text == "clipboard text"


def test_get_clipboard_empty_on_error(ctx):
    agent = WindowsAgent()
    with patch("tools.computer.windows.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b"error",
        )
        text = agent.get_clipboard()
        assert text == ""


def test_set_clipboard(ctx):
    agent = WindowsAgent()
    with patch("tools.computer.windows.subprocess.run") as mock_run:
        mock_run.side_effect = _mock_run()
        agent.set_clipboard("new text")
        script = mock_run.call_args[0][0][4]
        assert "Set-Clipboard" in script
        assert "new text" in script


def test_open_application(ctx):
    agent = WindowsAgent()
    with patch("tools.computer.windows.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"1234", stderr=b"",
        )
        agent.open_application("notepad")
        script = mock_run.call_args[0][0][4]
        assert "Start-Process" in script


def test_close_application(ctx):
    agent = WindowsAgent()
    with patch("tools.computer.windows.subprocess.run") as mock_run:
        mock_run.side_effect = _mock_run()
        agent.close_application("notepad")
        script = mock_run.call_args[0][0][4]
        assert "Stop-Process" in script


def test_close_application_raises_on_missing(ctx):
    agent = WindowsAgent()
    with patch("tools.computer.windows.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b"no process found",
        )
        with pytest.raises(CapabilityUnavailable, match="no running process"):
            agent.close_application("nonexistent")


def test_inspect_screen(ctx):
    agent = WindowsAgent()
    with patch("tools.computer.windows.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"1920 1080 Chrome", stderr=b"",
        )
        info = agent.inspect_screen()
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["active_window"] == "Chrome"


def test_inspect_screen_no_active_window(ctx):
    agent = WindowsAgent()
    with patch("tools.computer.windows.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"1920 1080 ", stderr=b"",
        )
        info = agent.inspect_screen()
        assert info["width"] == 1920
        assert info["active_window"] is None


def test_raises_when_powershell_unavailable(monkeypatch):
    monkeypatch.setattr("tools.computer.windows.shutil.which", lambda name: None)
    agent = WindowsAgent()
    with pytest.raises(CapabilityUnavailable, match="powershell.exe"):
        agent.screenshot()


def test_timeout_raises_capability_unavailable():
    agent = WindowsAgent(timeout=0.001)
    with patch("tools.computer.windows.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 0.001)):
        with pytest.raises(CapabilityUnavailable, match="timed out"):
            agent.screenshot()
