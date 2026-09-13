"""Actual Tk widget checks when a Tcl/Tk display is available."""
import queue
import tkinter as tk
from types import SimpleNamespace

import pytest

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
