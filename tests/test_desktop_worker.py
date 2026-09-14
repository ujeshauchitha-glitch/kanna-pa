import queue
import threading
from types import SimpleNamespace

import pytest

from core.agent.loop import AgentLoop
from core.bootstrap import build_registry
from core.config.settings import Settings
from core.events.bus import EventBus
from core.memory.db import Database
from core.permissions.sandbox import Sandbox
from core.planner.rule_based import RuleBasedPlanner
from core.tools.context import ToolContext
from interfaces.desktop.worker import DesktopWorker, DesktopGate, compose_request


def receive(worker, kind):
    for _ in range(100):
        event, payload = worker.events.get(timeout=5)
        if event == kind:
            return payload
    pytest.fail(f"did not receive {kind}")


def factory(tmp_path, threads, **kwargs):
    import logging
    def build(*, gate):
        threads.append(threading.get_ident())
        db = Database(tmp_path / "desktop.db")
        db.migrate()
        events = EventBus()
        registry = build_registry(gate=gate)
        settings = Settings(llm_provider="rule_based")
        sandbox = Sandbox([tmp_path])
        planner = kwargs.get("planner", RuleBasedPlanner())
        def loop(*, session_id):
            threads.append(threading.get_ident())
            ctx = ToolContext(db, settings, sandbox, events, logging.getLogger("test"), session_id)
            return AgentLoop(registry, planner, ctx, max_corrections=0)
        def close():
            threads.append(threading.get_ident())
            db.close()
        return SimpleNamespace(db=db, event_bus=events, registry=registry, settings=settings,
                               sandbox=sandbox, planner=planner, agent_loop=loop, close=close)
    return build


def test_real_agent_session_runs_and_closes_on_one_thread(tmp_path):
    threads = []
    worker = DesktopWorker(factory=factory(tmp_path, threads))
    assert not worker.submit("list files in .")
    worker.start()
    ready = receive(worker, "ready")
    # The task history dialog queries this same handle directly from the
    # Tk thread — Database is explicitly thread-safe (its own RLock,
    # check_same_thread=False), so sharing the reference is intentional.
    assert isinstance(ready["db"], Database)
    assert worker.submit("list files in .")
    result = receive(worker, "result")
    assert result["state"] == "complete"
    receive(worker, "idle")
    worker.close()
    worker.thread.join(5)
    assert not worker.thread.is_alive()
    assert len(set(threads)) == 1
    assert threads[0] != threading.get_ident()
    with Database(tmp_path / "desktop.db") as db:
        assert db.query_one("SELECT ended_at FROM sessions")["ended_at"]
        assert [r["role"] for r in db.query("SELECT role FROM messages ORDER BY id")] == ["user", "assistant"]


@pytest.mark.parametrize("allow", [False, True])
def test_desktop_approval_controls_real_overwrite_and_blocks_overlap(tmp_path, allow):
    from core.planner.plan import Plan, PlanStep
    path = tmp_path / "existing.txt"
    path.write_text("original")
    planner = SimpleNamespace(create_plan=lambda *a: Plan("overwrite", [PlanStep("fs_write_file",
                {"path": str(path), "content": "changed", "overwrite": True})]))
    worker = DesktopWorker(factory=factory(tmp_path, [], planner=planner))
    worker.start()
    receive(worker, "ready")
    assert worker.submit("overwrite")
    approval = receive(worker, "approval")
    assert not worker.submit("overlapping")
    assert approval.args["path"] == str(path)
    approval.respond(allow)
    result = receive(worker, "result")
    assert result["state"] == ("complete" if allow else "failed")
    assert path.read_text() == ("changed" if allow else "original")
    assert result["files"] == ([str(path)] if allow else [])
    worker.close()
    worker.thread.join(5)
    assert not worker.thread.is_alive()


def test_worker_cancel_denies_pending_approval_and_reports_cancelled(tmp_path):
    from core.planner.plan import Plan, PlanStep
    path = tmp_path / "existing.txt"
    path.write_text("original")
    planner = SimpleNamespace(create_plan=lambda *a: Plan("overwrite", [PlanStep("fs_write_file",
                {"path": str(path), "content": "changed", "overwrite": True})]))
    worker = DesktopWorker(factory=factory(tmp_path, [], planner=planner))
    worker.start()
    receive(worker, "ready")
    assert worker.submit("overwrite")
    receive(worker, "approval")  # left unanswered — cancelled instead
    assert worker.cancel()
    result = receive(worker, "result")
    assert result["state"] == "cancelled"
    assert path.read_text() == "original"  # never actually overwritten
    receive(worker, "idle")
    worker.close()
    worker.thread.join(5)


def test_cancel_is_a_noop_when_nothing_is_running(tmp_path):
    worker = DesktopWorker(factory=factory(tmp_path, []))
    worker.start()
    receive(worker, "ready")
    assert worker.cancel() is False
    worker.close()
    worker.thread.join(5)


def test_closing_releases_pending_approval_as_denied():
    events, stop, cancel = queue.Queue(), threading.Event(), threading.Event()
    gate = DesktopGate(events, stop, cancel)
    answers = []
    thread = threading.Thread(target=lambda: answers.append(gate.approve(
        tool_name="delete", args={}, level=None, reason="review")))
    thread.start()
    _kind, request = events.get(timeout=2)
    assert _kind == "approval"
    stop.set()
    thread.join(2)
    assert answers == [False]
    # The UI needs to know to close the (otherwise orphaned) dialog it
    # opened for this exact request.
    assert events.get(timeout=2) == ("approval_resolved", request)


def test_cancelling_releases_pending_approval_as_denied_without_stopping():
    events, stop, cancel = queue.Queue(), threading.Event(), threading.Event()
    gate = DesktopGate(events, stop, cancel)
    answers = []
    thread = threading.Thread(target=lambda: answers.append(gate.approve(
        tool_name="delete", args={}, level=None, reason="review")))
    thread.start()
    assert events.get(timeout=2)[0] == "approval"
    cancel.set()
    thread.join(2)
    assert answers == [False]
    assert not stop.is_set()  # cancelling one task must not shut down the worker


def test_startup_failure_does_not_accept_work():
    def fail(**kwargs):
        raise RuntimeError("configuration unavailable")
    worker = DesktopWorker(factory=fail)
    worker.start()
    assert receive(worker, "startup_error") == "configuration unavailable"
    worker.thread.join(2)
    assert not worker.submit("anything")


def test_attachment_request_keeps_exact_paths_and_requires_intent():
    import json
    paths = ['C:/workspace/my "report".txt']
    request = compose_request("Summarize", paths)
    assert json.loads(request.split("\n")[-1]) == paths
    with pytest.raises(ValueError):
        compose_request(" ", paths)


def test_failed_request_does_not_poison_following_run(tmp_path):
    worker = DesktopWorker(factory=factory(tmp_path, []))
    worker.start()
    receive(worker, "ready")
    assert worker.submit("read missing.txt")
    assert receive(worker, "result")["state"] == "failed"
    receive(worker, "idle")
    assert worker.submit("list files in .")
    assert receive(worker, "result")["state"] == "complete"
    worker.close()
    worker.thread.join(5)
