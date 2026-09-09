from __future__ import annotations

from core.memory.repositories.execution_log import ExecutionLogRepository
from core.memory.repositories.sessions import SessionRepository
from core.memory.repositories.tasks import TaskRepository
from core.tools.result import ToolResult


def test_session_repository_crud(db):
    repo = SessionRepository(db)
    session = repo.create({"source": "test"})
    assert repo.get(session.id).id == session.id

    repo.add_message(session.id, "user", "hello")
    repo.add_message(session.id, "assistant", "hi there")
    messages = repo.get_messages(session.id)
    assert [m.role for m in messages] == ["user", "assistant"]

    repo.end(session.id)
    assert repo.get(session.id).ended_at is not None


def test_task_repository_crud(db):
    repo = TaskRepository(db)
    task = repo.create("Write report", project="school")
    assert task.status == "pending"

    repo.set_status(task.id, "in_progress")
    assert repo.get(task.id).status == "in_progress"

    repo.set_status(task.id, "completed")
    assert [t.id for t in repo.list(status="completed")] == [task.id]
    assert repo.list(status="pending") == []


def test_execution_log_repository_records_and_lists(db):
    repo = ExecutionLogRepository(db)
    result = ToolResult.ok({"a": 1})
    result.duration_ms = 12.5
    repo.record(session_id="s1", tool_name="fs_read_file", args={"path": "x"}, result=result)

    recent = repo.recent()
    assert len(recent) == 1
    assert recent[0]["tool_name"] == "fs_read_file"

    for_session = repo.for_session("s1")
    assert len(for_session) == 1
