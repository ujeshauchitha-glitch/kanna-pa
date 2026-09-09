"""The context object passed to every tool's `execute()`.

Bundles the shared services a tool might need so tools don't each import
and construct their own DB connection, config, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from logging import Logger

from core.config.settings import Settings
from core.events.bus import EventBus
from core.memory.db import Database
from core.permissions.sandbox import Sandbox


@dataclass
class ToolContext:
    db: Database
    settings: Settings
    sandbox: Sandbox
    event_bus: EventBus
    logger: Logger
    session_id: str | None = None
