"""Explicit agent states.

`AgentLoop` transitions through these in order (with CORRECTING looping
back to EXECUTING when a step fails and a retry budget remains) and
persists every transition, so a run is inspectable after the fact rather
than being an opaque black box.
"""
from __future__ import annotations

import enum


class AgentState(enum.Enum):
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    SELECTING = "selecting"
    EXECUTING = "executing"
    OBSERVING = "observing"
    CHECKING = "checking"
    CORRECTING = "correcting"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"
    BLOCKED = "blocked"


TERMINAL_STATES = frozenset({AgentState.COMPLETE, AgentState.FAILED, AgentState.BLOCKED})
