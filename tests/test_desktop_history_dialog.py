"""Real Tk widget checks for the task history dialog — skipped where no
Tcl/Tk display can start, same convention as test_desktop_ui.py.
"""
from __future__ import annotations

import tkinter as tk

import pytest

from core.memory.db import Database
from core.memory.repositories.plans import PlanRepository
from core.memory.repositories.sessions import SessionRepository
from core.planner.plan import Plan, PlanStep
from interfaces.desktop.history_dialog import HistoryDialog

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


@pytest.fixture
def db():
    d = Database(":memory:")
    d.migrate()
    yield d
    d.close()


def _seed_run(db, *, request, kept_path, missing_path=None):
    session = SessionRepository(db).create({"source": "desktop"})
    plan = Plan(request, [PlanStep("fs_write_file", {"path": kept_path, "content": "hi"})])
    repo = PlanRepository(db)
    plan_id = repo.create(plan, session_id=session.id)
    from core.tools.result import ToolResult
    files = [kept_path] + ([missing_path] if missing_path else [])
    result = ToolResult.ok({}, files_created=files).to_dict()
    result["attempts"] = [{"attempt": 1}]
    repo.record_step_result(plan_id, 0, status="ok", result=result)
    repo.set_status(plan_id, "complete")
    return plan_id


def test_empty_history_shows_explanation(root, db):
    dialog = HistoryDialog(root, db, PALETTE)
    try:
        assert dialog.runs == []
        assert "No task history yet" in dialog.detail.get("1.0", "end-1c")
    finally:
        dialog.top.destroy()


def test_run_selection_shows_steps_and_files(root, db, tmp_path):
    kept = tmp_path / "kept.txt"
    kept.write_text("hi")
    missing = tmp_path / "moved.txt"  # recorded but never actually created here
    _seed_run(db, request="write a report", kept_path=str(kept), missing_path=str(missing))

    dialog = HistoryDialog(root, db, PALETTE)
    try:
        assert len(dialog.runs) == 1
        assert dialog.runs_list.size() == 1
        text = dialog.detail.get("1.0", "end-1c")
        assert "write a report" in text
        assert "fs_write_file" in text

        listed = [dialog.files_list.get(i) for i in range(dialog.files_list.size())]
        assert any(entry.startswith("✓") and "kept.txt" in entry for entry in listed)
        assert any(entry.startswith("✗ missing") and "moved.txt" in entry for entry in listed)
    finally:
        dialog.top.destroy()


def test_refresh_preserves_selection(root, db):
    # The file need not exist on disk for this test — only selection
    # persistence across _reload() is under test here.
    plan_id = _seed_run(db, request="first run", kept_path="placeholder.txt")
    dialog = HistoryDialog(root, db, PALETTE)
    try:
        assert dialog._selected_run_id() == plan_id
        _seed_run(db, request="second run", kept_path="placeholder.txt")
        dialog._reload()
        assert dialog._selected_run_id() == plan_id  # still on the run we had open
        assert len(dialog.runs) == 2
    finally:
        dialog.top.destroy()


def test_copy_path_puts_selected_file_on_clipboard(root, db, tmp_path):
    kept = tmp_path / "kept.txt"
    kept.write_text("hi")
    _seed_run(db, request="write a report", kept_path=str(kept))

    dialog = HistoryDialog(root, db, PALETTE)
    try:
        dialog.files_list.selection_set(0)
        dialog._copy_selected_path()
        assert dialog.top.clipboard_get() == str(kept)
    finally:
        dialog.top.destroy()
