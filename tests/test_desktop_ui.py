"""Actual Tk widget checks when a Tcl/Tk display is available."""
import queue
import tkinter as tk
from types import SimpleNamespace

import pytest

from core.memory.db import Database
from interfaces.desktop.app import KannaApp


@pytest.fixture
def app():
    worker = SimpleNamespace(events=queue.Queue(), start=lambda: None, submit=lambda text: True,
                             cancel=lambda: True)
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


def test_cancel_button_enabled_only_while_busy(app):
    assert app.cancel_btn.cget("state") == "disabled"
    app.worker.events.put(("ready", {"planner": "RuleBasedPlanner", "model": "none", "roots": ["."], "tools": 1}))
    app._poll()
    app._draft("list files in .")
    app._on_send()
    assert app.cancel_btn.cget("state") == "normal"
    app.worker.events.put(("result", {"state": "cancelled", "message": "Cancelled after 0 of 1 step(s) completed.", "files": []}))
    app.worker.events.put(("idle", None))
    app._poll()
    assert app.cancel_btn.cget("state") == "disabled"
    assert app.status.cget("text") == "Cancelled"


def test_clicking_cancel_calls_worker_cancel_and_disables_itself(app):
    calls = []
    app.worker.cancel = lambda: (calls.append(1) or True)
    app.worker.events.put(("ready", {"planner": "RuleBasedPlanner", "model": "none", "roots": ["."], "tools": 1}))
    app._poll()
    app._draft("list files in .")
    app._on_send()
    app._on_cancel()
    assert calls == [1]
    assert app.cancel_btn.cget("state") == "disabled"
    assert "Cancelling" in app.status.cget("text")


def test_cancel_click_when_worker_has_nothing_to_cancel_is_harmless(app):
    app.worker.cancel = lambda: False
    app._on_cancel()  # must not raise even though nothing is running


def test_pending_approval_dialog_closes_when_gate_resolves_it_first(app):
    from interfaces.desktop.worker import ApprovalRequest
    request = ApprovalRequest("fs_delete", {"path": "x"}, "irreversible")
    app.worker.events.put(("approval", request))
    app._poll()
    assert app._approval_dialog is not None
    assert app._approval_dialog.winfo_exists()

    app.worker.events.put(("approval_resolved", request))
    app._poll()
    assert app._approval_dialog is None
    assert app._approval_request is None


def _button_labeled(widget, text):
    for child in widget.winfo_children():
        if isinstance(child, tk.Button) and child.cget("text") == text:
            return child
        found = _button_labeled(child, text)
        if found is not None:
            return found
    return None


def test_approval_dialog_cancel_task_button_denies_and_requests_cancellation(app):
    from interfaces.desktop.worker import ApprovalRequest
    calls = []
    app.worker.cancel = lambda: (calls.append(1) or True)
    request = ApprovalRequest("fs_write_file", {"path": "x"}, "overwrite")
    app.worker.events.put(("approval", request))
    app._poll()

    # The dialog is application-modal (grab_set) — the main window's own
    # Cancel button is unreachable while it's open, so this must be
    # reachable *from the dialog itself*.
    button = _button_labeled(app._approval_dialog, "Cancel task")
    assert button is not None
    button.invoke()

    assert calls == [1]
    assert request.answered.is_set()
    assert request.allowed is False
    assert app._approval_dialog is None


def test_approval_resolved_for_a_different_request_leaves_open_dialog_alone(app):
    from interfaces.desktop.worker import ApprovalRequest
    open_request = ApprovalRequest("fs_delete", {"path": "x"}, "irreversible")
    other_request = ApprovalRequest("fs_delete", {"path": "y"}, "irreversible")
    app.worker.events.put(("approval", open_request))
    app._poll()
    dialog = app._approval_dialog
    assert dialog is not None

    app.worker.events.put(("approval_resolved", other_request))
    app._poll()
    assert app._approval_dialog is dialog
    assert dialog.winfo_exists()
    dialog.destroy()


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
