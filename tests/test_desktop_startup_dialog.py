"""Real Tk widget checks for the startup-at-login dialog — skipped where
no Tcl/Tk display can start, same convention as the other desktop UI
tests. The backend itself is a small in-memory fake here (already
covered for real in test_desktop_startup.py); this file is about the
dialog's own state machine and button wiring.
"""
from __future__ import annotations

import tkinter as tk

import pytest

PALETTE = {"BG": "#10151d", "PANEL": "#19212d", "FIELD": "#111923", "TEXT": "#e7edf5",
           "MUTED": "#a5b3c6", "ACCENT": "#9ce6cf", "ERROR": "#ffb0b0"}


class _FakeBackend:
    def __init__(self, *, supported=True, enabled=False, enable_error=None, disable_error=None):
        self.supported = supported
        self.enabled = enabled
        self.enable_error = enable_error
        self.disable_error = disable_error
        self.enable_calls = 0
        self.disable_calls = 0

    def is_supported(self):
        return self.supported

    def unsupported_reason(self):
        return "Not supported on this fake platform."

    def mechanism_description(self):
        return "a fake per-user mechanism"

    def is_enabled(self):
        return self.enabled

    def enable(self):
        self.enable_calls += 1
        if self.enable_error:
            return self.enable_error
        self.enabled = True
        return None

    def disable(self):
        self.disable_calls += 1
        if self.disable_error:
            return self.disable_error
        self.enabled = False
        return None


@pytest.fixture
def root():
    try:
        r = tk.Tk()
        r.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"Tk runtime/display unavailable: {exc}")
    yield r
    r.destroy()


def _open(root, backend):
    from interfaces.desktop.startup_dialog import StartupDialog
    return StartupDialog(root, PALETTE, backend=backend)


def test_unsupported_backend_disables_both_buttons(root):
    dialog = _open(root, _FakeBackend(supported=False))
    try:
        assert dialog.enable_btn.cget("state") == "disabled"
        assert dialog.disable_btn.cget("state") == "disabled"
        assert "Not supported" in dialog.status.cget("text")
    finally:
        dialog.top.destroy()


def test_supported_and_disabled_shows_enable_only(root):
    dialog = _open(root, _FakeBackend(supported=True, enabled=False))
    try:
        assert dialog.enable_btn.cget("state") == "normal"
        assert dialog.disable_btn.cget("state") == "disabled"
        assert "Disabled" in dialog.status.cget("text")
    finally:
        dialog.top.destroy()


def test_clicking_enable_calls_backend_and_flips_to_enabled_state(root):
    backend = _FakeBackend(supported=True, enabled=False)
    dialog = _open(root, backend)
    try:
        dialog.enable_btn.invoke()
        assert backend.enable_calls == 1
        assert dialog.enable_btn.cget("state") == "disabled"
        assert dialog.disable_btn.cget("state") == "normal"
        assert "Enabled" in dialog.status.cget("text")
    finally:
        dialog.top.destroy()


def test_clicking_disable_calls_backend_and_flips_to_disabled_state(root):
    backend = _FakeBackend(supported=True, enabled=True)
    dialog = _open(root, backend)
    try:
        dialog.disable_btn.invoke()
        assert backend.disable_calls == 1
        assert dialog.enable_btn.cget("state") == "normal"
        assert dialog.disable_btn.cget("state") == "disabled"
    finally:
        dialog.top.destroy()


def test_enable_failure_shows_error_and_state_stays_disabled(root, monkeypatch):
    shown = {}
    monkeypatch.setattr(
        "interfaces.desktop.startup_dialog.messagebox.showerror",
        lambda title, message, **kw: shown.update(title=title, message=message))
    backend = _FakeBackend(supported=True, enabled=False, enable_error="permission denied")
    dialog = _open(root, backend)
    try:
        dialog.enable_btn.invoke()
        assert shown["message"] == "permission denied"
        # A failed enable must not be reported as if it succeeded.
        assert dialog.enable_btn.cget("state") == "normal"
        assert "Disabled" in dialog.status.cget("text")
    finally:
        dialog.top.destroy()
