"""Hotkey spec parsing and platform dispatch are pure/deterministic and
tested directly. The X11 backend's actual grab/deliver behavior is real
OS integration and is exercised for real against a live X11 session,
separately, at the bottom of this file — skipped where none exists.
"""
from __future__ import annotations

import subprocess
import threading
import time

import pytest

from interfaces.desktop.hotkey import (
    DEFAULT_HOTKEY, NullHotkeyBackend, WindowsHotkeyBackend, X11HotkeyBackend,
    configured_hotkey, create_hotkey_backend, parse_hotkey,
)

# --- parse_hotkey ---------------------------------------------------------


def test_parse_default_hotkey():
    mods, key = parse_hotkey(DEFAULT_HOTKEY)
    assert mods == frozenset({"control", "shift"})
    assert key == "k"


def test_parse_hotkey_normalizes_aliases_and_case():
    mods, key = parse_hotkey("Cmd+Win+Meta+Super+X")
    assert mods == frozenset({"super"})
    assert key == "x"


@pytest.mark.parametrize("spec", ["k", "ctrl+", "ctrl+shift+kk", "ctrl+shift+-", "banana+k"])
def test_parse_hotkey_rejects_malformed_specs(spec):
    with pytest.raises(ValueError):
        parse_hotkey(spec)


# --- configured_hotkey -----------------------------------------------------


def test_configured_hotkey_defaults(monkeypatch):
    monkeypatch.delenv("KANNA_DESKTOP_HOTKEY", raising=False)
    assert configured_hotkey() == DEFAULT_HOTKEY


def test_configured_hotkey_reads_env(monkeypatch):
    monkeypatch.setenv("KANNA_DESKTOP_HOTKEY", "ctrl+alt+j")
    assert configured_hotkey() == "ctrl+alt+j"


# --- create_hotkey_backend dispatch (no real registration attempted) ------


def test_invalid_configured_hotkey_yields_explained_null_backend(monkeypatch):
    backend = create_hotkey_backend(lambda: None, spec="not-a-hotkey")
    assert isinstance(backend, NullHotkeyBackend)
    reason = backend.register()
    assert "KANNA_DESKTOP_HOTKEY" in reason and "not-a-hotkey" in reason


def test_windows_platform_dispatches_to_windows_backend(monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    backend = create_hotkey_backend(lambda: None)
    assert isinstance(backend, WindowsHotkeyBackend)


def test_darwin_platform_dispatches_to_explained_null_backend(monkeypatch):
    monkeypatch.setattr("sys.platform", "darwin")
    backend = create_hotkey_backend(lambda: None)
    assert isinstance(backend, NullHotkeyBackend)
    assert "macOS" in backend.reason


def test_wayland_session_dispatches_to_explained_null_backend(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    backend = create_hotkey_backend(lambda: None)
    assert isinstance(backend, NullHotkeyBackend)
    assert "Wayland" in backend.reason
    # Never claims an X11 mechanism works here.
    assert "X11" in backend.reason


def test_x11_session_dispatches_to_x11_backend(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    backend = create_hotkey_backend(lambda: None)
    assert isinstance(backend, X11HotkeyBackend)


def test_unset_session_type_dispatches_to_x11_backend(monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    backend = create_hotkey_backend(lambda: None)
    assert isinstance(backend, X11HotkeyBackend)


def test_null_backend_poll_and_unregister_are_harmless_noops():
    backend = NullHotkeyBackend("unavailable for testing")
    backend.poll()
    backend.unregister()  # must not raise


# --- Live X11 integration: real XGrabKey against a real X server ----------


def _x11_hotkey_available() -> bool:
    try:
        from Xlib import display as xdisplay
    except ImportError:
        return False
    try:
        xdisplay.Display().close()
    except Exception:
        return False
    return subprocess.run(["which", "xdotool"], capture_output=True).returncode == 0


pytestmark_live = pytest.mark.skipif(
    not _x11_hotkey_available(), reason="no live X11 session + python-xlib + xdotool")


@pytestmark_live
def test_x11_backend_fires_on_the_real_key_combo():
    triggered = threading.Event()
    backend = X11HotkeyBackend(frozenset({"control", "shift"}), "j", triggered.set)
    reason = backend.register()
    assert reason is None, reason
    try:
        subprocess.run(["xdotool", "key", "ctrl+shift+j"], check=True)
        deadline = time.time() + 2.0
        while time.time() < deadline and not triggered.is_set():
            backend.poll()
            time.sleep(0.05)
        assert triggered.is_set()
    finally:
        backend.unregister()


@pytestmark_live
def test_x11_backend_ignores_unrelated_keys():
    triggered = threading.Event()
    backend = X11HotkeyBackend(frozenset({"control", "shift"}), "j", triggered.set)
    reason = backend.register()
    assert reason is None, reason
    try:
        subprocess.run(["xdotool", "key", "a"], check=True)
        time.sleep(0.3)
        backend.poll()
        assert not triggered.is_set()
    finally:
        backend.unregister()
