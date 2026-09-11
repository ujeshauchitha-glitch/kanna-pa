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
from core.llm.anthropic_provider import AnthropicProvider
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.registry import ToolRegistry
from core.tools.result import ToolResult
from core.tools.schema import array, integer, obj, string
from tools.authoring.base import AuthoredContent
from tools.authoring.llm_author import LLMAuthor

_SECTION_SCHEMA = obj({
    "heading": string(default=""),
    "level": integer(default=1),
    "paragraphs": array(string()),
    "bullets": array(string()),
})

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
            "source_text": string(description="The source material to author from (text content from a file, extraction, etc.)"),
            "task": string(description="What to produce: e.g. 'Answer the assignment questions', 'Write a summary report'"),
            "title": string(default="", description="Optional document title; the model chooses if omitted"),
        },
        required=("source_text", "task"),
    )
    output_schema = _AUTHORED_SCHEMA

    def __init__(self, author=None) -> None:
        self._author = author

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        source_text = args.get("source_text", "")
        task = args.get("task", "")
        title = args.get("title") or None

        if not source_text.strip():
            return ToolResult.fail("empty_source", "source_text is empty; provide material to author from")
        if not task.strip():
            return ToolResult.fail("empty_task", "task is empty; describe what to produce")

        author = self._author
        if author is None:
            try:
                provider = AnthropicProvider(
                    model=ctx.settings.llm_model, max_tokens=ctx.settings.llm_max_tokens,
                )
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


ALL_TOOLS = [AuthorContentTool()]


def register_all(registry: ToolRegistry) -> None:
    for tool in ALL_TOOLS:
        registry.register(tool)
