"""The task system.

A `Task` is a unit of work Kanna is tracking on the user's behalf —
distinct from a `Plan`/`PlanStep` (one agent run's execution steps) and
from `finance` records. This service is the entry point the CLI and
agent tools use instead of touching `TaskRepository` directly.
"""
from __future__ import annotations

from core.memory.db import Database
from core.memory.repositories.tasks import TaskRepository
from core.tasks.models import Task


class TaskService:
    def __init__(self, db: Database) -> None:
        self._repo = TaskRepository(db)

    def create(self, title: str, *, description: str | None = None, project: str | None = None,
               due_date: str | None = None) -> Task:
        return self._repo.create(title, description=description, project=project, due_date=due_date)

    def start(self, task_id: str) -> None:
        self._repo.set_status(task_id, "in_progress")

    def complete(self, task_id: str) -> None:
        self._repo.set_status(task_id, "completed")

    def cancel(self, task_id: str) -> None:
        self._repo.set_status(task_id, "cancelled")

    def get(self, task_id: str) -> Task | None:
        return self._repo.get(task_id)

    def list(self, *, status: str | None = None) -> list[Task]:
        return self._repo.list(status=status)
