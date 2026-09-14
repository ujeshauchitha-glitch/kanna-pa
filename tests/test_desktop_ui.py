"""Actual Tk widget checks when a Tcl/Tk display is available."""
import queue
import tkinter as tk
from types import SimpleNamespace

import pytest

from core.memory.db import Database
from interfaces.desktop.app import KannaApp


@pytest.fixture
def app():
    worker = SimpleNamespace(events=queue.Queue(), start=lambda: None, submit=lambda text: True)
    try:
        app = KannaApp(worker=worker, integrations=False)
    except tk.TclError as exc:
        pytest.skip(f"Tk runtime/display unavailable: {exc}")
    yield app
    app.root.destroy()


def test_ready_draft_and_submit(app):
    app.worker.events.put(("ready", {"planner": "RuleBasedPlanner", "model": "none", "roots": ["."], "tools": 1}))
    app._poll()
    app._draft("list files in .")
    app._on_send()
    assert app.busy
    assert app.send_btn.cget("state") == "disabled"
    app.worker.events.put(("result", {"state": "failed", "message": "Actual failure", "files": []}))
    app.worker.events.put(("idle", None))
    app._poll()
    assert app.status.cget("text") == "Task failed"
    assert not app.busy
    assert app.entry.get("1.0", "end-1c") == "list files in ."


def test_voice_only_fills_draft(app):
    app.worker.events.put(("transcript", "Review these sources"))
    app._poll()
    assert app.entry.get("1.0", "end-1c") == "Review these sources"
    assert not app.busy


def test_minimum_window_keeps_composer_visible(app):
    app.root.geometry("860x640")
    app.root.update()
    assert app.entry.winfo_width() > 400
    assert app.send_btn.winfo_ismapped()
    assert app.artifacts.winfo_ismapped()
    assert app.output.winfo_height() >= 80


def test_history_button_opens_dialog_once_db_is_ready(app):
    db = Database(":memory:")
    db.migrate()
    try:
        app.worker.events.put(("ready", {"planner": "RuleBasedPlanner", "model": "none",
                                          "roots": ["."], "tools": 1, "db": db}))
        app._poll()
        assert app.db is db
        app._show_history()  # must not raise even with an empty history
    finally:
        if app.history_dialog is not None:
            app.history_dialog.top.destroy()
        db.close()


def test_history_button_reuses_open_dialog_instead_of_duplicating(app):
    db = Database(":memory:")
    db.migrate()
    try:
        app.worker.events.put(("ready", {"planner": "RuleBasedPlanner", "model": "none",
                                          "roots": ["."], "tools": 1, "db": db}))
        app._poll()
        app._show_history()
        first = app.history_dialog
        app._show_history()
        assert app.history_dialog is first
    finally:
        if app.history_dialog is not None:
            app.history_dialog.top.destroy()
        db.close()


def test_history_button_before_ready_shows_a_message_not_a_crash(app, monkeypatch):
    shown = {}
    monkeypatch.setattr(
        "interfaces.desktop.app.messagebox.showinfo",
        lambda title, message, **kw: shown.update(title=title, message=message))
    assert app.db is None
    app._show_history()
    assert "starting" in shown.get("message", "").lower()


def test_real_hotkey_registration_never_raises_and_cleans_up():
    """integrations=True exercises the real create_hotkey_backend() wiring
    (not the fixture's integrations=False path). On a live X11 session
    this actually grabs the configured combo; anywhere else it logs an
    explained NullHotkeyBackend reason instead — either way, construction
    and teardown must never raise."""
    worker = SimpleNamespace(events=queue.Queue(), start=lambda: None, submit=lambda text: True)
    try:
        app = KannaApp(worker=worker, integrations=True)
    except tk.TclError as exc:
        pytest.skip(f"Tk runtime/display unavailable: {exc}")
    try:
        assert app._hotkey_backend is not None
    finally:
        app._destroy()
