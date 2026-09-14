"""Read-only history of past runs, built entirely from evidence the agent
loop already persists (`plans`/`plan_steps` via `PlanRepository`, joined
to `sessions` for the entry point that made the request) — no new schema,
no rerun, and no dependency on anything still being in memory.

A run here is one plan: one submitted request, its steps, their status,
attempts, and the files they actually produced. A file's continued
existence is checked live and reported honestly — a path recorded when
the step ran may since have been moved or deleted, and this says so
rather than pretending otherwise or crashing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.memory.db import Database


def _exists(path: str) -> bool:
    try:
        return Path(path).is_file()
    except OSError:
        return False


@dataclass
class RunFile:
    path: str
    exists: bool


@dataclass
class RunStep:
    tool_name: str
    status: str  # "ok" | "failed" | "pending" (never reached — an earlier step failed first)
    attempts: int
    message: str
    files: list[RunFile] = field(default_factory=list)


@dataclass
class RunSummary:
    id: str
    session_id: str | None
    request: str
    status: str
    created_at: str
    updated_at: str
    step_count: int
    failed_step_count: int


@dataclass
class RunDetail(RunSummary):
    steps: list[RunStep] = field(default_factory=list)


def list_runs(db: Database, *, source: str | None = "desktop", limit: int = 200) -> list[RunSummary]:
    """Most recent first. `source` filters to sessions whose metadata
    tags them with this entry point (desktop sessions by default, since
    that's the only UI with a history browser today) — pass `source=None`
    to include every run regardless of where it was submitted from.
    """
    rows = db.query(
        "SELECT p.id, p.session_id, p.request, p.status, p.created_at, p.updated_at, "
        "s.metadata AS session_metadata "
        "FROM plans p LEFT JOIN sessions s ON s.id = p.session_id "
        "ORDER BY p.created_at DESC LIMIT ?",
        (limit,),
    )
    summaries = []
    for row in rows:
        if source is not None:
            meta = json.loads(row["session_metadata"]) if row["session_metadata"] else {}
            if meta.get("source") != source:
                continue
        steps = db.query("SELECT status FROM plan_steps WHERE plan_id = ?", (row["id"],))
        failed = sum(1 for s in steps if s["status"] == "failed")
        summaries.append(RunSummary(
            id=row["id"], session_id=row["session_id"], request=row["request"],
            status=row["status"], created_at=row["created_at"], updated_at=row["updated_at"],
            step_count=len(steps), failed_step_count=failed,
        ))
    return summaries


def get_run(db: Database, plan_id: str) -> RunDetail | None:
    """Full detail for one run — every step's status, message, and the
    files it produced, with live existence checks. Returns None if the
    plan id is unknown (e.g. a stale id from a since-cleared database)."""
    plan_row = db.query_one("SELECT * FROM plans WHERE id = ?", (plan_id,))
    if plan_row is None:
        return None
    step_rows = db.query(
        "SELECT * FROM plan_steps WHERE plan_id = ? ORDER BY step_index ASC", (plan_id,)
    )
    steps: list[RunStep] = []
    failed = 0
    for row in step_rows:
        result = json.loads(row["result"]) if row["result"] else {}
        attempts = result.get("attempts", [])
        seen: set[str] = set()
        files: list[RunFile] = []
        for path in [*result.get("files_created", []), *result.get("files_modified", [])]:
            if path in seen:
                continue
            seen.add(path)
            files.append(RunFile(path=path, exists=_exists(path)))
        error = result.get("error")
        if row["status"] == "failed":
            failed += 1
            message = error["message"] if error else "Failed (no error detail recorded)."
        elif row["status"] == "ok":
            data = result.get("data", {})
            note = data.get("message")
            message = note if isinstance(note, str) else "Completed."
        elif plan_row["status"] == "cancelled":
            message = "Not run — the task was cancelled first."
        else:
            message = "Not run — an earlier step failed first."
        steps.append(RunStep(tool_name=row["tool_name"], status=row["status"],
                              attempts=len(attempts), message=message, files=files))
    return RunDetail(
        id=plan_row["id"], session_id=plan_row["session_id"], request=plan_row["request"],
        status=plan_row["status"], created_at=plan_row["created_at"], updated_at=plan_row["updated_at"],
        step_count=len(steps), failed_step_count=failed, steps=steps,
    )
