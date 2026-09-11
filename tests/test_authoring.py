"""Tests for the source-aware content authoring tool.

Uses FakeProvider to script LLM responses; no network required.
Tests cover: successful authoring, empty source, empty task,
LLM unavailable, malformed LLM response, schema validation,
end-to-end workflow (read -> author -> generate PDF), and
provenance/unresolved tracking.
"""
from __future__ import annotations

import json

import pytest

from core.agent.loop import AgentLoop
from core.agent.state import AgentState
from core.bootstrap import build_registry
from core.llm.base import LLMResponse
from core.llm.fake import FakeProvider
from core.permissions.gate import PreApprovedGate
from core.planner.llm_planner import LLMPlanner
from core.tools.result import ToolResult
from tools.authoring.base import AuthoredContent, AuthoredSection
from tools.authoring.llm_author import LLMAuthor, _parse_response
from tools.authoring.tools import AuthorContentTool


def _fake_authored_response(title="Test Report", sections=None, unresolved=None):
    """Build a scripted LLM response for the authoring tool."""
    payload = {
        "title": title,
        "sections": sections or [
            {"heading": "Introduction", "level": 1,
             "paragraphs": ["This report summarizes the source material."], "bullets": []},
        ],
        "unresolved": unresolved or [],
    }
    return LLMResponse(content=json.dumps(payload))


def _make_tool(responses=None):
    """Create an AuthorContentTool backed by a FakeProvider."""
    provider = FakeProvider(responses or [_fake_authored_response()])
    return AuthorContentTool(author=LLMAuthor(provider)), provider


def _authoring_registry(responses=None):
    """Build a registry with author_content backed by a FakeProvider."""
    registry = build_registry()
    tool, _ = _make_tool(responses)
    registry._tools["author_content"] = tool
    return registry


# --- Unit tests for the tool ---


def test_author_content_success(ctx):
    tool, _ = _make_tool()
    result = tool.execute({
        "source_text": "Kanna is an AI agent that executes work.",
        "task": "Summarize what Kanna is.",
    }, ctx)
    assert result.success
    assert result.data["title"] == "Test Report"
    assert len(result.data["sections"]) >= 1
    assert result.data["sections"][0]["heading"] == "Introduction"
    assert "This report summarizes the source material." == result.data["sections"][0]["paragraphs"][0]
    assert isinstance(result.data["unresolved"], list)
    assert isinstance(result.data["provenance"], list)


def test_author_content_empty_source(ctx):
    tool, _ = _make_tool()
    result = tool.execute({"source_text": "", "task": "Summarize"}, ctx)
    assert not result.success
    assert result.error.code == "empty_source"


def test_author_content_whitespace_source(ctx):
    tool, _ = _make_tool()
    result = tool.execute({"source_text": "   \n  ", "task": "Summarize"}, ctx)
    assert not result.success
    assert result.error.code == "empty_source"


def test_author_content_empty_task(ctx):
    tool, _ = _make_tool()
    result = tool.execute({"source_text": "Some source text.", "task": ""}, ctx)
    assert not result.success
    assert result.error.code == "empty_task"


def test_author_content_whitespace_task(ctx):
    tool, _ = _make_tool()
    result = tool.execute({"source_text": "Some source text.", "task": "  "}, ctx)
    assert not result.success
    assert result.error.code == "empty_task"


def test_author_content_malformed_llm_response(ctx):
    provider = FakeProvider([LLMResponse(content="not json at all")])
    tool = AuthorContentTool(author=LLMAuthor(provider))
    result = tool.execute({
        "source_text": "Some source.",
        "task": "Summarize.",
    }, ctx)
    assert not result.success
    assert result.error.code == "authoring_failed"


def test_author_content_provenance_records_source_and_task(ctx):
    tool, _ = _make_tool()
    result = tool.execute({
        "source_text": "The quick brown fox.",
        "task": "Write a report about foxes.",
        "title": "Fox Report",
    }, ctx)
    assert result.success
    provenance = result.data["provenance"]
    assert any("20 characters" in p for p in provenance)
    assert any("foxes" in p for p in provenance)
    assert any("Fox Report" in p for p in provenance)


def test_author_content_unresolved_questions_preserved(ctx):
    provider = FakeProvider([_fake_authored_response(
        title="Assignment",
        sections=[{"heading": "Q1", "level": 1,
                   "paragraphs": ["Partially answered."], "bullets": []}],
        unresolved=["Could not determine the molecular formula from the source."],
    )])
    tool = AuthorContentTool(author=LLMAuthor(provider))
    result = tool.execute({
        "source_text": "Partial chemistry notes.",
        "task": "Answer the assignment questions.",
    }, ctx)
    assert result.success
    assert result.data["unresolved"] == ["Could not determine the molecular formula from the source."]


def test_author_content_tool_registered_in_bootstrap(ctx):
    registry = build_registry()
    assert registry.has("author_content")
    tool = registry.get("author_content")
    assert tool.permission is not None


# --- Unit tests for _parse_response ---


def test_parse_response_valid():
    content = json.dumps({
        "title": "Parsed Report",
        "sections": [{"heading": "H1", "level": 1, "paragraphs": ["para"], "bullets": ["b1"]}],
        "unresolved": ["question?"],
    })
    result = _parse_response(content, "source text", "task", "Parsed Report")
    assert isinstance(result, AuthoredContent)
    assert result.title == "Parsed Report"
    assert len(result.sections) == 1
    assert result.sections[0].heading == "H1"
    assert result.sections[0].paragraphs == ["para"]
    assert result.sections[0].bullets == ["b1"]
    assert result.unresolved == ["question?"]
    assert any("11 characters" in p for p in result.provenance)
    assert any("task" in p for p in result.provenance)
    assert any("Parsed Report" in p for p in result.provenance)


def test_parse_response_no_json():
    with pytest.raises(ValueError, match="did not return JSON"):
        _parse_response("hello world", "src", "task", None)


def test_parse_response_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_response("{bad json}", "src", "task", None)


def test_parse_response_non_dict():
    with pytest.raises(ValueError, match="did not return JSON|must be an object"):
        _parse_response('[1, 2, 3]', "src", "task", None)


def test_parse_response_empty_sections():
    content = json.dumps({"title": "Empty", "sections": [], "unresolved": []})
    result = _parse_response(content, "src", "task", None)
    assert result.title == "Empty"
    assert result.sections == []


def test_parse_response_skips_malformed_sections():
    content = json.dumps({
        "title": "Mixed",
        "sections": [
            "not a dict",
            {"heading": "Good", "level": 1, "paragraphs": ["ok"], "bullets": []},
        ],
        "unresolved": [],
    })
    result = _parse_response(content, "src", "task", None)
    assert len(result.sections) == 1
    assert result.sections[0].heading == "Good"


def test_parse_response_title_fallback():
    content = json.dumps({"title": "", "sections": [], "unresolved": []})
    result = _parse_response(content, "src", "task", "Fallback Title")
    assert result.title == "Fallback Title"


# --- Integration: end-to-end read -> author -> generate PDF ---


def test_end_to_end_read_author_generate_pdf(ctx, tmp_path):
    pytest.importorskip("reportlab")
    (tmp_path / "notes.txt").write_text("Kanna is a personal AI work-execution agent.")

    result = run_plan(ctx, [
        step("fs_read_file", {"path": "notes.txt"}, data_equals={"truncated": False}),
        step("author_content", {
            "source_text": ref("0.data.content"),
            "task": "Write a one-paragraph summary.",
        }, data_nonempty_key="title"),
        step("document_generate_pdf", {
            "path": "summary.pdf",
            "title": ref("1.data.title"),
            "sections": ref("1.data.sections"),
        }, files_exist=["summary.pdf"], min_files_created=1),
    ])
    assert result.state == AgentState.COMPLETE
    assert (tmp_path / "summary.pdf").read_bytes().startswith(b"%PDF-")


def test_end_to_end_read_author_generate_docx(ctx, tmp_path):
    pytest.importorskip("docx")
    (tmp_path / "notes.txt").write_text("The experiment measured temperature over time.")

    result = run_plan(ctx, [
        step("fs_read_file", {"path": "notes.txt"}),
        step("author_content", {
            "source_text": ref("0.data.content"),
            "task": "Write an experiment report.",
            "title": "Lab Report",
        }, data_nonempty_key="title"),
        step("document_generate_docx", {
            "path": "report.docx",
            "title": ref("1.data.title"),
            "sections": ref("1.data.sections"),
        }, files_exist=["report.docx"], min_files_created=1),
    ])
    assert result.state == AgentState.COMPLETE
    assert (tmp_path / "report.docx").exists()


# --- Helpers for plan-based tests ---


def run_plan(ctx, steps, registry=None, corrections=2):
    provider = FakeProvider([LLMResponse(content=json.dumps({"steps": steps}))])
    reg = registry or _authoring_registry()
    return AgentLoop(reg, LLMPlanner(provider), ctx,
                     max_corrections=corrections).run("Execute this workflow")


def step(tool, args, **expected):
    return {"tool_name": tool, "args": args, "expected": expected or {"success": True}}


def ref(path):
    return {"$ref": path}
