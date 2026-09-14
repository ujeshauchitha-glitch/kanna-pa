"""Task system domain types.

Re-exports the persistence-layer `TaskRecord` under the name `Task` for
the service layer's public API, so callers depend on `core.tasks`
without reaching into `core.memory.repositories`.
"""
from __future__ import annotations

from core.memory.repositories.tasks import TaskRecord as Task

__all__ = ["Task"]
