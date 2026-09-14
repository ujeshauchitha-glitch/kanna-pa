"""core.agent.history reads exactly what AgentLoop already persists to
plans/plan_steps — no rerun, no new schema. Runs here are generated with
the real AgentLoop against real filesystem tools so files_created and
failures are genuine, not simulated.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

from core.agent.history import get_run, list_runs
from core.agent.loop import AgentLoop
from core.memory.repositories.sessions import SessionRepository
from core.planner.plan import Plan, PlanStep
from core.tools.context import ToolContext


def _run(db, sandbox, settings, event_bus, registry, *, source, plan):
    session = SessionRepository(db).create({"source": source})
    ctx = ToolContext(db=db, settings=settings, sandbox=sandbox, event_bus=event_bus,
                       logger=logging.getLogger("kanna.test"), session_id=session.id)
    planner = SimpleNamespace(create_plan=lambda *a: plan)
    return AgentLoop(registry, planner, ctx, max_corrections=0).run(plan.request)


def test_list_runs_filters_by_source_and_orders_recent_first(db, sandbox, settings, event_bus, registry):
    desktop_plan = Plan("desktop task", [PlanStep("fs_write_file",
        {"path": "a.txt", "content": "hi", "overwrite": False})])
    other_plan = Plan("cli task", [PlanStep("fs_write_file",
        {"path": "b.txt", "content": "hi", "overwrite": False})])
    _run(db, sandbox, settings, event_bus, registry, source="test", plan=other_plan)
    _run(db, sandbox, settings, event_bus, registry, source="desktop", plan=desktop_plan)

    desktop_only = list_runs(db)
    assert [r.request for r in desktop_only] == ["desktop task"]

    everything = list_runs(db, source=None)
    assert [r.request for r in everything] == ["desktop task", "cli task"]  # most recent first


def test_get_run_reports_real_files_and_missing_ones_honestly(db, sandbox, settings, event_bus, registry, tmp_path):
    plan = Plan("write a file", [PlanStep("fs_write_file",
        {"path": "kept.txt", "content": "hi", "overwrite": False})])
    result = _run(db, sandbox, settings, event_bus, registry, source="desktop", plan=plan)
    assert result.state.value == "complete"

    plan_id = db.query_one("SELECT id FROM plans ORDER BY created_at DESC LIMIT 1")["id"]
    detail = get_run(db, plan_id)
    assert detail.status == "complete"
    assert len(detail.steps) == 1
    [file] = detail.steps[0].files
    assert file.path.endswith("kept.txt")
    assert file.exists is True

    # The file is later moved/deleted outside Kanna's knowledge — history
    # must report that honestly, not keep claiming it's there.
    (tmp_path / "kept.txt").unlink()
    assert get_run(db, plan_id).steps[0].files[0].exists is False


def test_get_run_reports_failed_step_with_real_error_message(db, sandbox, settings, event_bus, registry):
    plan = Plan("read something missing", [PlanStep("fs_read_file", {"path": "missing.txt"})])
    result = _run(db, sandbox, settings, event_bus, registry, source="desktop", plan=plan)
    assert result.state.value == "failed"

    plan_id = db.query_one("SELECT id FROM plans ORDER BY created_at DESC LIMIT 1")["id"]
    detail = get_run(db, plan_id)
    assert detail.status == "failed"
    assert detail.failed_step_count == 1
    assert detail.steps[0].status == "failed"
    assert detail.steps[0].message  # a real message, not empty
    assert detail.steps[0].files == []


def test_get_run_returns_none_for_unknown_plan_id(db):
    assert get_run(db, "no-such-plan") is None


def test_list_runs_step_and_failure_counts(db, sandbox, settings, event_bus, registry):
    plan = Plan("two steps, one fails", [
        PlanStep("fs_write_file", {"path": "ok.txt", "content": "hi", "overwrite": False}),
        PlanStep("fs_read_file", {"path": "still-missing.txt"}),
    ])
    _run(db, sandbox, settings, event_bus, registry, source="desktop", plan=plan)
    [summary] = list_runs(db)
    assert summary.step_count == 2
    assert summary.failed_step_count == 1
