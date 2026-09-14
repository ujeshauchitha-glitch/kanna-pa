"""Startup-at-login backends — deterministic, no real registry/autostart
directory ever touched. The Windows backend is tested by injecting a
fake `winreg` module (the real one doesn't exist on non-Windows, and
even on Windows a real test must never write the user's actual Startup
registry entry); the Linux backend is tested against a `tmp_path`
XDG_CONFIG_HOME, never the real `~/.config/autostart`.
"""
from __future__ import annotations

import sys

import pytest

from interfaces.desktop.startup import (
    LinuxAutostartBackend, NullStartupBackend, WindowsStartupBackend,
    _quote_desktop_exec, _quote_windows, create_startup_backend,
)

# --- quoting (pure functions) ----------------------------------------------


def test_quote_windows_wraps_only_parts_with_spaces():
    assert _quote_windows(["C:\\Python\\python.exe", "C:\\a b\\main.py", "app"]) == \
        'C:\\Python\\python.exe "C:\\a b\\main.py" app'


def test_quote_desktop_exec_escapes_reserved_characters():
    quoted = _quote_desktop_exec(["/usr/bin/python3", "/home/me/a $b/main.py", "app"])
    assert quoted == '/usr/bin/python3 "/home/me/a \\$b/main.py" app'


# --- create_startup_backend dispatch ----------------------------------------


def test_windows_platform_dispatches_to_windows_backend(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert isinstance(create_startup_backend(), WindowsStartupBackend)


def test_linux_platform_dispatches_to_linux_backend(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert isinstance(create_startup_backend(), LinuxAutostartBackend)


def test_darwin_platform_dispatches_to_explained_null_backend(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    backend = create_startup_backend()
    assert isinstance(backend, NullStartupBackend)
    assert not backend.is_supported()
    assert "platform" in backend.unsupported_reason().lower()
    assert backend.enable() == backend.unsupported_reason()
    assert backend.disable() is None  # nothing to remove — trivially fine


# --- Windows backend, via a fake winreg -------------------------------------


class _FakeKey:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeWinreg:
    HKEY_CURRENT_USER = object()
    KEY_SET_VALUE = 1
    REG_SZ = 1

    def __init__(self):
        self.values: dict[str, str] = {}

    def CreateKeyEx(self, hive, path, reserved, access):
        return _FakeKey()

    def OpenKey(self, hive, path, *args):
        return _FakeKey()

    def SetValueEx(self, key, name, reserved, value_type, value):
        self.values[name] = value

    def QueryValueEx(self, key, name):
        if name not in self.values:
            raise FileNotFoundError(name)
        return (self.values[name], self.REG_SZ)

    def DeleteValue(self, key, name):
        if name not in self.values:
            raise FileNotFoundError(name)
        del self.values[name]


@pytest.fixture
def fake_winreg(monkeypatch):
    fake = _FakeWinreg()
    monkeypatch.setitem(sys.modules, "winreg", fake)
    return fake


def test_windows_backend_starts_disabled(fake_winreg):
    assert WindowsStartupBackend().is_enabled() is False


def test_windows_backend_enable_then_disable_round_trip(fake_winreg):
    backend = WindowsStartupBackend()
    assert backend.enable() is None
    assert backend.is_enabled() is True
    assert "Kanna" in fake_winreg.values
    assert "main.py" in fake_winreg.values["Kanna"]
    assert "app" in fake_winreg.values["Kanna"]

    assert backend.disable() is None
    assert backend.is_enabled() is False


def test_windows_backend_disable_when_never_enabled_is_a_harmless_noop(fake_winreg):
    assert WindowsStartupBackend().disable() is None


def test_windows_backend_enable_reports_missing_main_py(fake_winreg, monkeypatch):
    monkeypatch.setattr("interfaces.desktop.startup._launch_command_parts", lambda: None)
    reason = WindowsStartupBackend().enable()
    assert reason is not None and "main.py" in reason
    assert not fake_winreg.values


def test_windows_backend_reports_registry_write_failure(fake_winreg, monkeypatch):
    def raising_create_key(*a, **kw):
        raise OSError("access denied")
    monkeypatch.setattr(fake_winreg, "CreateKeyEx", raising_create_key)
    reason = WindowsStartupBackend().enable()
    assert reason is not None and "access denied" in reason


# --- Linux backend, against a throwaway XDG_CONFIG_HOME ---------------------


@pytest.fixture
def linux_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return LinuxAutostartBackend(), tmp_path


def test_linux_backend_starts_disabled(linux_backend):
    backend, _ = linux_backend
    assert backend.is_enabled() is False


def test_linux_backend_enable_writes_a_real_autostart_file(linux_backend):
    backend, config_home = linux_backend
    assert backend.enable() is None
    assert backend.is_enabled() is True

    content = (config_home / "autostart" / "kanna.desktop").read_text()
    assert "[Desktop Entry]" in content
    assert "Type=Application" in content
    assert "Name=Kanna" in content
    assert "main.py" in content and " app" in content
    assert "X-GNOME-Autostart-enabled=true" in content


def test_linux_backend_disable_removes_the_file_and_is_idempotent(linux_backend):
    backend, _ = linux_backend
    backend.enable()
    assert backend.disable() is None
    assert backend.is_enabled() is False
    assert backend.disable() is None  # calling again must not raise


def test_linux_backend_enable_reports_missing_main_py(linux_backend, monkeypatch):
    backend, _ = linux_backend
    monkeypatch.setattr("interfaces.desktop.startup._launch_command_parts", lambda: None)
    reason = backend.enable()
    assert reason is not None and "main.py" in reason
    assert backend.is_enabled() is False
