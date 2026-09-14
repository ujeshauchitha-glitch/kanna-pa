"""A deterministic planner: regex intent matching -> a single-step plan.

Always available (no LLM, no network) — this is what `kanna ask` falls
back to when no LLM provider is configured, and what the agent-loop tests
run against so they never depend on the network.

Deliberately narrow: it recognizes a handful of common request shapes
(list/read/search files, log run compiled code, finance entry/query) and
raises `PlanningError` for anything else rather than guessing. An
`LLMPlanner` (core/planner/llm_planner.py) handles the open-ended case.
"""
from __future__ import annotations

import re

from core.errors import PlanningError
from core.planner.plan import Plan, PlanStep
from core.tools.registry import ToolRegistry

_LIST_DIR_RE = re.compile(
    r"\blist\b.*?\b(?:files?|contents?|directory|dir)\b.*?\b(?:in|of|from)\s+(?P<path>\S+)", re.IGNORECASE
)
_READ_FILE_RE = re.compile(r"\bread\b.*?\bfile\b\s+(?P<path>\S+)|\bread\b\s+(?P<path2>\S+)", re.IGNORECASE)
_SEARCH_RE = re.compile(
    r"\bsearch\b.*?\bfor\s+['\"]?(?P<pattern>[^'\"]+?)['\"]?\s+in\s+(?P<path>\S+)", re.IGNORECASE
)
_FINANCE_SPEND_RE = re.compile(r"\b(spent|paid|bought|purchased)\b", re.IGNORECASE)
_FINANCE_QUERY_RE = re.compile(r"\bhow much\b.*\b(spend|spent|spending)\b", re.IGNORECASE)


class RuleBasedPlanner:
    def create_plan(self, request: str, registry: ToolRegistry) -> Plan:
        text = request.strip()

        if _FINANCE_QUERY_RE.search(text) and registry.has("finance_query"):
            return Plan(
                request=request,
                rationale="Recognized a spending question.",
                steps=[PlanStep(tool_name="finance_query", args={"text": text},
                                 description="Answer the spending question")],
            )

        if _FINANCE_SPEND_RE.search(text) and registry.has("finance_add_transaction"):
            return Plan(
                request=request,
                rationale="Recognized a spending statement.",
                steps=[PlanStep(tool_name="finance_add_transaction", args={"text": text},
                                 description="Log the transaction")],
            )

        match = _LIST_DIR_RE.search(text)
        if match and registry.has("fs_list_directory"):
            return Plan(
                request=request,
                rationale="Recognized a request to list a directory.",
                steps=[PlanStep(tool_name="fs_list_directory", args={"path": match.group("path")},
                                 description=f"List {match.group('path')}")],
            )

        match = _SEARCH_RE.search(text)
        if match and registry.has("fs_search_files"):
            return Plan(
                request=request,
                rationale="Recognized a file search request.",
                steps=[PlanStep(
                    tool_name="fs_search_files",
                    args={"path": match.group("path"), "pattern": match.group("pattern").strip()},
                    description=f"Search for '{match.group('pattern').strip()}' in {match.group('path')}",
                )],
            )

        match = _READ_FILE_RE.search(text)
        if match and registry.has("fs_read_file"):
            path = match.group("path") or match.group("path2")
            return Plan(
                request=request,
                rationale="Recognized a request to read a file.",
                steps=[PlanStep(tool_name="fs_read_file", args={"path": path},
                                 description=f"Read {path}")],
            )

        raise PlanningError(
            f"could not understand the request well enough to build a plan: {request!r}. "
            "Try phrasing it as a file operation (list/read/search) or a finance statement/question, "
            "or configure an LLM provider for open-ended requests."
        )
