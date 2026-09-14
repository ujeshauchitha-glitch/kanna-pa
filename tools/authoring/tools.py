"""Source-aware content authoring exposed through the tool registry.

`author_content` takes source material and a task description, uses the
LLM to produce structured Document-compatible content (title + sections
with headings/paragraphs/bullets), and returns the result with provenance
and explicit unresolved questions.  The output feeds directly into the
existing `documents` renderers via plan references.

This is the missing piece between "read a source" and "generate a
document" — it bridges vision/text extraction and document generation
with LLM-driven content authoring that is honest about what it cannot
answer.
"""
from __future__ import annotations

from core.errors import LLMUnavailable, SandboxViolation
from core.llm.factory import build_provider
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.registry import ToolRegistry
from core.tools.result import ToolResult
from core.tools.schema import OneOfSchema, array, integer, obj, string
from tools.authoring.assignment import solve_assignment
from tools.authoring.base import AuthoredContent
from tools.authoring.llm_author import LLMAuthor

_SECTION_SCHEMA = obj({
    "heading": string(default=""),
    "level": integer(default=1),
    "paragraphs": array(string()),
    "bullets": array(string()),
})


def _coerce_source_text(value) -> str:
    """Convert source_text to a string, handling list-of-sections from $ref."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for section in value:
            if isinstance(section, dict):
                heading = section.get("heading", "")
                if heading:
                    parts.append(f"## {heading}")
                for p in section.get("paragraphs", []):
                    parts.append(p)
                for b in section.get("bullets", []):
                    parts.append(f"- {b}")
            else:
                parts.append(str(section))
        return "\n".join(parts)
    return str(value)

_AUTHORED_SCHEMA = obj({
    "title": string(),
    "sections": array(_SECTION_SCHEMA),
    "unresolved": array(string()),
    "provenance": array(string()),
})


class AuthorContentTool:
    """Uses the LLM to author structured content from source material and a task.

    The output sections are directly compatible with the document
    generation tools' `sections` parameter — pass them via plan
    references to create a DOCX/PPTX/PDF without re-describing content.
    """

    name = "author_content"
    description = ("Read source material and a task description, then produce structured "
                   "Document-compatible content (title + sections) authored from the source. "
                   "The output feeds into document generation tools via plan references.")
    permission = PermissionLevel.LOW
    input_schema = obj(
        {
            "source_text": OneOfSchema(
                string(description="The source material as text"),
                array(_SECTION_SCHEMA, description="Structured sections from document_read"),
                description="Source material: text string or list of sections from document_read",
            ),
            "task": string(description="What to produce: e.g. 'Answer the assignment questions', 'Write a summary report'"),
            "title": string(default="", description="Optional document title; the model chooses if omitted"),
        },
        required=("source_text", "task"),
    )
    output_schema = _AUTHORED_SCHEMA

    def __init__(self, author=None) -> None:
        self._author = author

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        source_text = _coerce_source_text(args.get("source_text", ""))
        task = args.get("task", "")
        title = args.get("title") or None

        if not source_text.strip():
            return ToolResult.fail("empty_source", "source_text is empty; provide material to author from")
        if not task.strip():
            return ToolResult.fail("empty_task", "task is empty; describe what to produce")

        author = self._author
        if author is None:
            try:
                provider = build_provider(ctx.settings)
                author = LLMAuthor(provider)
            except LLMUnavailable as exc:
                return ToolResult.fail("llm_unavailable",
                                       f"content authoring requires an LLM provider: {exc}")

        try:
            result = author.author(source_text=source_text, task=task, title=title)
        except Exception as exc:  # noqa: BLE001 — provider failure is one reportable error
            return ToolResult.fail("authoring_failed", f"content authoring failed: {exc}")

        return ToolResult.ok(_to_result_dict(result))


def _to_result_dict(content: AuthoredContent) -> dict:
    """Convert AuthoredContent to a dict matching the output schema."""
    return {
        "title": content.title,
        "sections": [
            {
                "heading": s.heading or "",
                "level": s.level,
                "paragraphs": list(s.paragraphs),
                "bullets": list(s.bullets),
            }
            for s in content.sections
        ],
        "unresolved": list(content.unresolved),
        "provenance": list(content.provenance),
    }


_ANSWER_SCHEMA = obj({
    "question_id": string(),
    "question_text": string(),
    "answer_text": string(),
    "source_references": array(string()),
    "unresolved": array(string()),
})

_ASSIGNMENT_SCHEMA = obj({
    "title": string(),
    "questions_extracted": integer(),
    "questions_answered": integer(),
    "sections": array(_SECTION_SCHEMA),
    "answers": array(_ANSWER_SCHEMA),
    "unresolved": array(string()),
    "provenance": array(string()),
})


class AssignmentSolveTool:
    """Extracts questions from source material and answers each one.

    Uses the LLM to first identify individual questions/tasks from an
    assignment or document, then answers each one using only the provided
    source material.  Returns structured Document-compatible content with
    per-question answers, provenance, and explicit unresolved items.
    """

    name = "assignment_solve"
    description = ("Extract individual questions from source material and answer each one. "
                   "Returns structured content with per-question answers, provenance, "
                   "and unresolved items. Output feeds into document generation tools.")
    permission = PermissionLevel.LOW
    input_schema = obj(
        {
            "source_text": OneOfSchema(
                string(description="The assignment source material as text"),
                array(_SECTION_SCHEMA, description="Structured sections from document_read"),
                description="Assignment source: text string or list of sections from document_read",
            ),
            "task": string(description="What to do: e.g. 'Answer all questions in this assignment'"),
            "title": string(default="", description="Optional assignment title; the model chooses if omitted"),
        },
        required=("source_text", "task"),
    )
    output_schema = _ASSIGNMENT_SCHEMA

    def __init__(self, provider=None) -> None:
        self._provider = provider

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        source_text = _coerce_source_text(args.get("source_text", ""))
        task = args.get("task", "")
        title = args.get("title") or None

        if not source_text.strip():
            return ToolResult.fail("empty_source", "source_text is empty; provide the assignment material")
        if not task.strip():
            return ToolResult.fail("empty_task", "task is empty; describe what to solve")

        provider = self._provider
        if provider is None:
            try:
                provider = build_provider(ctx.settings)
            except LLMUnavailable as exc:
                return ToolResult.fail("llm_unavailable",
                                       f"assignment solving requires an LLM provider: {exc}")

        try:
            result = solve_assignment(provider, source_text, task)
        except Exception as exc:  # noqa: BLE001 — provider failure is one reportable error
            return ToolResult.fail("assignment_failed", f"assignment solving failed: {exc}")

        # Override title if provided.
        if title:
            result.title = title

        # Build sections from answers — each question becomes a section.
        sections = []
        for answer in result.answers:
            sections.append({
                "heading": f"{answer.question_id}: {answer.question_text[:80]}",
                "level": 1,
                "paragraphs": [answer.answer_text] if answer.answer_text else [],
                "bullets": [],
            })

        return ToolResult.ok({
            "title": result.title,
            "questions_extracted": len(result.questions),
            "questions_answered": len(result.answers),
            "sections": sections,
            "answers": [
                {
                    "question_id": a.question_id,
                    "question_text": a.question_text,
                    "answer_text": a.answer_text,
                    "source_references": list(a.source_references),
                    "unresolved": list(a.unresolved),
                }
                for a in result.answers
            ],
            "unresolved": list(result.unresolved),
            "provenance": list(result.provenance),
        })


ALL_TOOLS = [AuthorContentTool(), AssignmentSolveTool()]


def register_all(registry: ToolRegistry) -> None:
    for tool in ALL_TOOLS:
        registry.register(tool)
