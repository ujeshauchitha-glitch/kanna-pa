from __future__ import annotations

import logging

import pytest

from core.config.settings import Settings
from core.events.bus import EventBus
from core.memory.db import Database
from core.memory.repositories.sessions import SessionRepository
from core.permissions.gate import DenyAllGate, PreApprovedGate
from core.permissions.sandbox import Sandbox
from core.tools.context import ToolContext
from core.tools.registry import ToolRegistry
from documents import reader as document_reader
from finance import tools as finance_tools
from tools import filesystem, process


@pytest.fixture
def db() -> Database:
    database = Database(":memory:")
    database.migrate()
    yield database
    database.close()


@pytest.fixture
def sandbox(tmp_path):
    return Sandbox([str(tmp_path)])


@pytest.fixture
def settings() -> Settings:
    return Settings(default_currency="INR", timezone="UTC")


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def ctx(db, sandbox, settings, event_bus) -> ToolContext:
    session = SessionRepository(db).create({"source": "test"})
    return ToolContext(db=db, settings=settings, sandbox=sandbox, event_bus=event_bus,
                        logger=logging.getLogger("kanna.test"), session_id=session.id)


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry(gate=PreApprovedGate({"fs_write_file", "fs_delete"}))
    filesystem.register_all(reg)
    process.register_all(reg)
    finance_tools.register_all(reg)
    document_reader.register_all(reg)
    return reg


@pytest.fixture
def strict_registry() -> ToolRegistry:
    """A registry whose gate denies everything — for testing that REVIEW actions are blocked."""
    reg = ToolRegistry(gate=DenyAllGate())
    filesystem.register_all(reg)
    process.register_all(reg)
    finance_tools.register_all(reg)
    document_reader.register_all(reg)
    return reg
