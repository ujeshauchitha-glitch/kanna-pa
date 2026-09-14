"""Real Tk widget checks for the connection settings dialog — skipped
where no Tcl/Tk display can start. Save is exercised against a real
tmp_path config file (via update_config_file, not mocked) so the test
proves the file actually ends up correct, not just that a function was
called.
"""
from __future__ import annotations

import tkinter as tk

import pytest

from core.config.settings import Settings, load_settings

PALETTE = {"BG": "#10151d", "PANEL": "#19212d", "FIELD": "#111923", "TEXT": "#e7edf5",
           "MUTED": "#a5b3c6", "ACCENT": "#9ce6cf", "ERROR": "#ffb0b0"}


@pytest.fixture
def root():
    try:
        r = tk.Tk()
        r.withdraw()
    except tk.TclError as exc:
        pytest.skip(f"Tk runtime/display unavailable: {exc}")
    yield r
    r.destroy()


def _open(root, info=None, settings=None):
    from interfaces.desktop.connection_dialog import ConnectionDialog
    return ConnectionDialog(root, info or {"planner": "RuleBasedPlanner", "tools": 42,
                                            "roots": ["/workspace"]}, settings, PALETTE)


def test_fields_prefilled_from_current_settings(root):
    settings = Settings(llm_provider="anthropic", llm_model="claude-sonnet-5",
                        llm_fallback_models="gpt-4o-mini", llm_timeout_seconds=90)
    dialog = _open(root, settings=settings)
    try:
        assert dialog.provider_var.get() == "anthropic"
        assert dialog.model_entry.get() == "claude-sonnet-5"
        assert dialog.fallback_entry.get() == "gpt-4o-mini"
        assert dialog.timeout_entry.get() == "90"
    finally:
        dialog.top.destroy()


def test_save_writes_a_real_config_file_and_confirms(root, monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr("interfaces.desktop.connection_dialog.paths.config_path", lambda: config_path)
    dialog = _open(root, settings=Settings())
    try:
        dialog.provider_var.set("anthropic")
        dialog.model_entry.delete(0, tk.END)
        dialog.model_entry.insert(0, "claude-sonnet-5")
        dialog.timeout_entry.delete(0, tk.END)
        dialog.timeout_entry.insert(0, "45")
        dialog._on_save()

        assert config_path.exists()
        saved = load_settings(config_path)
        assert saved.llm_provider == "anthropic"
        assert saved.llm_model == "claude-sonnet-5"
        assert saved.llm_timeout_seconds == 45
        assert "Saved" in dialog.feedback.cget("text")
        assert str(config_path) in dialog.feedback.cget("text")
    finally:
        dialog.top.destroy()


def test_invalid_timeout_is_rejected_and_nothing_is_written(root, monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr("interfaces.desktop.connection_dialog.paths.config_path", lambda: config_path)
    shown = {}
    monkeypatch.setattr(
        "interfaces.desktop.connection_dialog.messagebox.showerror",
        lambda title, message, **kw: shown.update(title=title, message=message))
    dialog = _open(root, settings=Settings())
    try:
        dialog.timeout_entry.delete(0, tk.END)
        dialog.timeout_entry.insert(0, "not-a-number")
        dialog._on_save()

        assert "positive whole number" in shown["message"]
        assert not config_path.exists()
        assert dialog.feedback.cget("text") == ""  # never claims success
    finally:
        dialog.top.destroy()


def test_empty_model_is_rejected(root, monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr("interfaces.desktop.connection_dialog.paths.config_path", lambda: config_path)
    shown = {}
    monkeypatch.setattr(
        "interfaces.desktop.connection_dialog.messagebox.showerror",
        lambda title, message, **kw: shown.update(title=title, message=message))
    dialog = _open(root, settings=Settings())
    try:
        dialog.model_entry.delete(0, tk.END)
        dialog._on_save()

        assert "model" in shown["message"].lower()
        assert not config_path.exists()
    finally:
        dialog.top.destroy()


def test_no_settings_yet_falls_back_to_sane_defaults_not_a_crash(root):
    dialog = _open(root, settings=None)
    try:
        assert dialog.provider_var.get()  # some non-empty default provider
        assert dialog.timeout_entry.get() == "120"
    finally:
        dialog.top.destroy()
