"""Wires every subsystem together into one `Kanna` instance.

This is the single place that constructs settings, the database, the
sandbox, the tool registry (with every Phase 1 tool registered and the
default permission rules applied), and the planner. The CLI and tests
both call `bootstrap()` instead of each re-assembling these pieces, so
there's exactly one definition of "how Kanna is wired".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from core.agent.loop import AgentLoop
from core.config import paths
from core.config.settings import Settings, load_settings
from core.errors import LLMUnavailable
from core.events.bus import EventBus
from core.llm.anthropic_provider import AnthropicProvider
from core.logging.setup import setup_logging
from core.memory.db import Database
from core.permissions.gate import ApprovalGate, DenyAllGate
from core.permissions.levels import Decision
from core.permissions.policy import PermissionPolicy, Rule
from core.permissions.sandbox import Sandbox
from core.planner.base import Planner
from core.planner.llm_planner import LLMPlanner
from core.planner.rule_based import RuleBasedPlanner
from core.tools.context import ToolContext
from core.tools.registry import ToolRegistry
from documents import tools as document_tools
from finance import tools as finance_tools
from tools import filesystem, process
from tools.browser import tools as browser_tools
from tools.computer import tools as computer_tools

# Tools registered at REVIEW by default (the safe default — see each
# tool's docstring) whose risk actually hinges on one thing: whether
# they'd overwrite an existing file. Creating a brand-new file is
# low-risk; only an explicit overwrite=True stays REVIEW-gated.
_CREATE_OR_OVERWRITE_TOOLS = frozenset({
    "fs_write_file", "document_generate_docx", "document_generate_pptx", "document_generate_pdf",
})


def default_policy() -> PermissionPolicy:
    """The default permission rule set.

    Every other REVIEW-level tool (delete, etc.) keeps the registry's
    default behavior — approval required, denied unless a gate says yes.
    """
    policy = PermissionPolicy()
    policy.add_rule(Rule(
        name="creating a new file is low-risk; overwriting an existing one stays gated",
        predicate=lambda name, args: name in _CREATE_OR_OVERWRITE_TOOLS and not args.get("overwrite", False),
        decision=Decision.ALLOW,
    ))
    return policy


def build_registry(*, gate: ApprovalGate | None = None,
                    policy: PermissionPolicy | None = None) -> ToolRegistry:
    registry = ToolRegistry(policy=policy or default_policy(), gate=gate or DenyAllGate())
    filesystem.register_all(registry)
    process.register_all(registry)
    finance_tools.register_all(registry)
    document_tools.register_all(registry)
    computer_tools.register_all(registry)
    browser_tools.register_all(registry)
    return registry


def build_planner(settings: Settings, *, force_rule_based: bool = False) -> Planner:
    rule_based = RuleBasedPlanner()
    if force_rule_based or settings.llm_provider != "anthropic":
        return rule_based

    try:
        provider = AnthropicProvider(model=settings.llm_model, max_tokens=settings.llm_max_tokens)
        provider._get_client()  # fail fast here rather than on first real request
    except LLMUnavailable:
        return rule_based
    return LLMPlanner(provider, fallback=rule_based)


@dataclass
class Kanna:
    settings: Settings
    db: Database
    sandbox: Sandbox
    event_bus: EventBus
    registry: ToolRegistry
    planner: Planner
    logger: logging.Logger

    def tool_context(self, *, session_id: str | None = None) -> ToolContext:
        return ToolContext(db=self.db, settings=self.settings, sandbox=self.sandbox,
                            event_bus=self.event_bus, logger=self.logger, session_id=session_id)

    def agent_loop(self, *, session_id: str | None = None) -> AgentLoop:
        return AgentLoop(self.registry, self.planner, self.tool_context(session_id=session_id),
                          max_corrections=self.settings.max_corrections)

    def close(self) -> None:
        self.db.close()


def bootstrap(*, config_path: Path | None = None, db_path: Path | str | None = None,
              gate: ApprovalGate | None = None, force_rule_based_planner: bool = False) -> Kanna:
    paths.ensure_dirs()
    settings = load_settings(config_path)
    logger = setup_logging(settings.log_level, paths.log_dir())

    db = Database(db_path if db_path is not None else paths.db_path())
    db.migrate()

    sandbox = Sandbox(list(settings.sandbox_roots))
    event_bus = EventBus()
    registry = build_registry(gate=gate)
    planner = build_planner(settings, force_rule_based=force_rule_based_planner)

    return Kanna(settings=settings, db=db, sandbox=sandbox, event_bus=event_bus,
                 registry=registry, planner=planner, logger=logger)
