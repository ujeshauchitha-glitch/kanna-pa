"""Tests for the Android phone computer-control backend.

`is_available`/`list_devices` are tested for real against the genuinely-
installed `adb` binary — this sandbox has no Android device attached, so
they're expected to report that honestly, and this test proves they do
without any mocking. Every `AdbPhoneAgent` method is tested against a
mocked `subprocess.run` (no real device exists to test against here —
see the very deliberate warning in tools/computer/phone.py's module
docstring about this backend's validation status).
"""
from __future__ import annotations

import subprocess

import pytest

from core.errors import CapabilityUnavailable
from tools.computer.base import Point
from tools.computer.phone import AdbPhoneAgent, is_available, list_devices


# --- real (device-less) availability detection -----------------------------


def test_is_available_is_false_with_no_real_device_attached():
    """Genuinely live: this sandbox has adb installed and no device
    attached — is_available() must say so honestly, not guess."""
    assert list_devices() == []
    assert is_available() is False


def test_is_available_false_when_adb_itself_is_missing(monkeypatch):
    monkeypatch.setattr("tools.computer.phone.shutil.which", lambda name: None)
    assert list_devices() == []
    assert is_available() is False


def test_list_devices_survives_a_broken_adb(monkeypatch):
    monkeypatch.setattr("tools.computer.phone.shutil.which", lambda name: "/usr/bin/adb")
    def raising(*a, **kw):
        raise OSError("adb crashed")
    monkeypatch.setattr("tools.computer.phone.subprocess.run", raising)
    assert list_devices() == []


# --- mocked-subprocess tests ------------------------------------------------


class _FakeAdb:
    """Answers `adb devices` with one fake device attached by default (so
    _require_adb()'s availability check passes without every single test
    needing to care about it), and returns queued responses to every
    other invocation in order."""

    def __init__(self, *, device_attached=True):
        self.calls: list[list[str]] = []
        self.device_attached = device_attached
        self._queue: list[subprocess.CompletedProcess] = []

    def queue(self, *, returncode=0, stdout=b"", stderr=b""):
        self._queue.append(subprocess.CompletedProcess([], returncode, stdout, stderr))

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        if cmd == ["adb", "devices"]:
            text = "List of devices attached\n" + ("EMULATOR001\tdevice\n" if self.device_attached else "")
            if kwargs.get("text"):
                return subprocess.CompletedProcess(cmd, 0, stdout=text, stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout=text.encode(), stderr=b"")
        if self._queue:
            queued = self._queue.pop(0)
            return subprocess.CompletedProcess(cmd, queued.returncode, queued.stdout, queued.stderr)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")


@pytest.fixture
def fake_adb(monkeypatch):
    fake = _FakeAdb()
    monkeypatch.setattr("tools.computer.phone.shutil.which", lambda name: "/usr/bin/adb")
    monkeypatch.setattr("tools.computer.phone.subprocess.run", fake)
    return fake


def _shell_calls(fake: _FakeAdb) -> list[str]:
    """The single command string sent to each `adb shell "..."` call."""
    return [c[2] for c in fake.calls if c[:2] == ["adb", "shell"]]


def test_screenshot_returns_the_real_png_bytes(fake_adb):
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    fake_adb.queue(stdout=png)
    assert AdbPhoneAgent().screenshot() == png
    assert fake_adb.calls[-1] == ["adb", "exec-out", "screencap", "-p"]


def test_screenshot_raises_cleanly_on_non_png_output(fake_adb):
    fake_adb.queue(returncode=1, stderr=b"screencap: not permitted")
    with pytest.raises(CapabilityUnavailable, match="did not return a PNG"):
        AdbPhoneAgent().screenshot()


def test_inspect_screen_parses_size_and_focused_package(fake_adb):
    fake_adb.queue(stdout=b"Physical size: 1080x2400\n")
    fake_adb.queue(stdout=(
        b"  mCurrentFocus=Window{a1b2c3d u0 com.android.chrome/"
        b"com.google.android.apps.chrome.Main}\n"
    ))
    info = AdbPhoneAgent().inspect_screen()
    assert info == {"width": 1080, "height": 2400, "active_window": "com.android.chrome"}


def test_inspect_screen_tolerates_missing_focus_line(fake_adb):
    fake_adb.queue(stdout=b"Physical size: 1080x2400\n")
    fake_adb.queue(stdout=b"nothing relevant here\n")
    info = AdbPhoneAgent().inspect_screen()
    assert info["width"] == 1080 and info["active_window"] is None


def test_inspect_screen_raises_on_unparseable_size(fake_adb):
    fake_adb.queue(stdout=b"nonsense")
    with pytest.raises(CapabilityUnavailable, match="wm size"):
        AdbPhoneAgent().inspect_screen()


def test_move_mouse_always_unavailable_no_adb_call_made(fake_adb):
    with pytest.raises(CapabilityUnavailable, match="cursor"):
        AdbPhoneAgent().move_mouse(Point(1, 1))
    assert not _shell_calls(fake_adb)


def test_tap_sends_input_tap(fake_adb):
    AdbPhoneAgent().click(Point(100, 200))
    assert _shell_calls(fake_adb) == ["input tap 100 200"]


def test_right_click_is_a_long_press_swipe(fake_adb):
    AdbPhoneAgent().click(Point(50, 60), button="right")
    assert _shell_calls(fake_adb) == ["input swipe 50 60 50 60 500"]


def test_click_rejects_unknown_button(fake_adb):
    with pytest.raises(ValueError, match="unknown mouse button"):
        AdbPhoneAgent().click(Point(0, 0), button="middle")
    assert not _shell_calls(fake_adb)


def test_type_text_quotes_the_argument_for_the_device_shell(fake_adb):
    AdbPhoneAgent().type_text("hello world & $(danger)")
    [call] = _shell_calls(fake_adb)
    assert call == "input text 'hello world & $(danger)'"


def test_key_press_maps_known_keys(fake_adb):
    AdbPhoneAgent().key_press("back")
    assert _shell_calls(fake_adb) == ["input keyevent KEYCODE_BACK"]


def test_key_press_uppercases_and_prefixes_unknown_keys(fake_adb):
    AdbPhoneAgent().key_press("a")
    assert _shell_calls(fake_adb) == ["input keyevent KEYCODE_A"]


def test_key_press_passes_through_an_already_prefixed_code(fake_adb):
    AdbPhoneAgent().key_press("KEYCODE_CAMERA")
    assert _shell_calls(fake_adb) == ["input keyevent KEYCODE_CAMERA"]


def test_scroll_swipes_up_for_positive_dy(fake_adb):
    fake_adb.queue(stdout=b"Physical size: 1000x2000\n")  # inspect_screen's wm size call
    fake_adb.queue(stdout=b"")  # inspect_screen's dumpsys window call
    AdbPhoneAgent().scroll(0, 5)
    calls = _shell_calls(fake_adb)
    assert calls[-1] == "input swipe 500 1000 500 700 200"  # scrolling down = content moves up


def test_scroll_swipes_left_for_positive_dx(fake_adb):
    fake_adb.queue(stdout=b"Physical size: 1000x2000\n")
    fake_adb.queue(stdout=b"")
    AdbPhoneAgent().scroll(5, 0)
    calls = _shell_calls(fake_adb)
    assert calls[-1] == "input swipe 500 1000 200 1000 200"


def test_scroll_zero_zero_makes_no_swipe_call(fake_adb):
    fake_adb.queue(stdout=b"Physical size: 1000x2000\n")
    fake_adb.queue(stdout=b"")
    AdbPhoneAgent().scroll(0, 0)
    assert not any("swipe" in c for c in _shell_calls(fake_adb))


def test_get_clipboard_always_unavailable_no_adb_call(fake_adb):
    with pytest.raises(CapabilityUnavailable, match="clipboard"):
        AdbPhoneAgent().get_clipboard()
    assert not _shell_calls(fake_adb)


def test_set_clipboard_always_unavailable_no_adb_call(fake_adb):
    with pytest.raises(CapabilityUnavailable, match="clipboard"):
        AdbPhoneAgent().set_clipboard("x")
    assert not _shell_calls(fake_adb)


def test_open_application_uses_package_id_directly(fake_adb):
    fake_adb.queue(stdout=b"", returncode=0)
    AdbPhoneAgent().open_application("com.android.chrome")
    assert _shell_calls(fake_adb) == [
        "monkey -p com.android.chrome -c android.intent.category.LAUNCHER 1"]


def test_open_application_resolves_a_fuzzy_name_via_pm_list(fake_adb):
    fake_adb.queue(stdout=b"package:com.android.chrome\n")  # pm list packages
    fake_adb.queue(stdout=b"")  # monkey launch
    AdbPhoneAgent().open_application("chrome")
    calls = _shell_calls(fake_adb)
    assert calls[0] == "pm list packages chrome"
    assert calls[1] == "monkey -p com.android.chrome -c android.intent.category.LAUNCHER 1"


def test_open_application_no_match_raises(fake_adb):
    fake_adb.queue(stdout=b"")  # pm list packages: nothing found
    with pytest.raises(CapabilityUnavailable, match="no installed package"):
        AdbPhoneAgent().open_application("nonexistent app")


def test_open_application_monkey_failure_raises(fake_adb):
    fake_adb.queue(stdout=b"No activities found to run, monkey aborted.\n", returncode=1)
    with pytest.raises(CapabilityUnavailable, match="could not launch"):
        AdbPhoneAgent().open_application("com.example.missing")


def test_close_application_force_stops_the_resolved_package(fake_adb):
    AdbPhoneAgent().close_application("com.android.chrome")
    assert _shell_calls(fake_adb) == ["am force-stop com.android.chrome"]


# --- availability gating ----------------------------------------------------


def test_raises_when_adb_binary_missing(monkeypatch):
    monkeypatch.setattr("tools.computer.phone.shutil.which", lambda name: None)
    with pytest.raises(CapabilityUnavailable, match="not on PATH"):
        AdbPhoneAgent().screenshot()


def test_raises_when_no_device_connected(monkeypatch):
    fake = _FakeAdb(device_attached=False)
    monkeypatch.setattr("tools.computer.phone.shutil.which", lambda name: "/usr/bin/adb")
    monkeypatch.setattr("tools.computer.phone.subprocess.run", fake)
    with pytest.raises(CapabilityUnavailable, match="no Android device is connected"):
        AdbPhoneAgent().screenshot()


def test_explicit_serial_skips_the_device_list_check(monkeypatch):
    """A caller who names a specific serial is trusted to know it's
    there — adb itself will fail the call if it isn't, same as any
    other adb error path."""
    fake = _FakeAdb(device_attached=False)
    monkeypatch.setattr("tools.computer.phone.shutil.which", lambda name: "/usr/bin/adb")
    monkeypatch.setattr("tools.computer.phone.subprocess.run", fake)
    fake.queue(stdout=b"Physical size: 100x200\n")
    fake.queue(stdout=b"")
    AdbPhoneAgent(serial="EMULATOR001").inspect_screen()
    assert fake.calls[0] == ["adb", "-s", "EMULATOR001", "shell", "wm size"]


def test_timeout_raises_capability_unavailable(monkeypatch):
    monkeypatch.setattr("tools.computer.phone.shutil.which", lambda name: "/usr/bin/adb")
    def raise_timeout(cmd, **kwargs):
        if cmd == ["adb", "devices"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="List of devices attached\nX\tdevice\n", stderr="")
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1))
    monkeypatch.setattr("tools.computer.phone.subprocess.run", raise_timeout)
    with pytest.raises(CapabilityUnavailable, match="timed out"):
        AdbPhoneAgent(timeout=0.01).screenshot()
