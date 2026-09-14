"""Vision capabilities exposed directly through the tool registry.

Unlike receipt/statement extraction (`finance/imports/receipt.py`,
`finance/imports/statement.py`) — where `vision.document` is a step
inside a `finance`-owned import pipeline with its own consumer-specific
tool — generic document structure has no fixed downstream consumer yet
(a future assignment-reading or report-rewriting workflow would use it,
per `docs/ROADMAP.md`), so it's registered here as its own read-only
tool instead of being buried inside another subsystem's tool.
"""
from __future__ import annotations

from core.errors import SandboxViolation, VisionUnavailable
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.registry import ToolRegistry
from core.tools.result import ToolResult
from core.tools.schema import array, integer, obj, string
from vision._common import guess_mime_type
from vision.document.anthropic_document import AnthropicDocumentProvider
from vision.document.base import DocumentProvider, DocumentStructure, StructureSection

_TABLE_SCHEMA = obj({"headers": array(string()), "rows": array(array(string()))})
_SECTION_SCHEMA = obj({
    "heading": string(), "level": integer(), "paragraphs": array(string()),
    "bullets": array(string()), "table": _TABLE_SCHEMA,
})


def _section_data(section: StructureSection) -> dict:
    data = {
        "heading": section.heading or "", "level": section.level,
        "paragraphs": list(section.paragraphs), "bullets": list(section.bullets),
    }
    # `table` is omitted (rather than set to null) when there is none —
    # `Schema.validate` only checks keys actually present, and "object,
    # but null" isn't a value its object/array types accept.
    if section.table is not None:
        data["table"] = {"headers": list(section.table.headers),
                          "rows": [list(r) for r in section.table.rows]}
    return data


def _structure_data(structure: DocumentStructure) -> dict:
    return {
        "title": structure.title or "", "sections": [_section_data(s) for s in structure.sections],
        "notes": structure.notes,
    }


class VisionExtractStructureTool:
    """Reads an arbitrary document's structure (title, sections, headings,
    paragraphs, bullets, tables) — not a fixed receipt or statement-row
    shape. Extraction only: it transcribes what's on the page, never
    summarizes or invents content (see `vision/document/anthropic_
    document.py`'s instruction prompt).
    """

    name = "vision_extract_structure"
    description = ("Read an arbitrary document image or PDF (path within the sandbox, may be "
                    "multi-page) and return its structure: title, sections with headings, "
                    "paragraphs, bullet lists, and tables.")
    permission = PermissionLevel.LOW
    input_schema = obj(
        {
            "path": string(description="Path to the document image/PDF, within the sandbox"),
            "mime_type": string(description="Override the guessed mime type, e.g. 'application/pdf'",
                                 default=""),
        },
        required=("path",),
    )
    output_schema = obj({"title": string(), "sections": array(_SECTION_SCHEMA), "notes": string()})

    def __init__(self, provider: DocumentProvider | None = None) -> None:
        self._provider = provider

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            resolved = ctx.sandbox.resolve(args["path"])
        except SandboxViolation as exc:
            return ToolResult.fail("sandbox_violation", str(exc))

        if not resolved.exists() or not resolved.is_file():
            return ToolResult.fail("not_found", f"file does not exist: {resolved}")

        mime_type = args.get("mime_type") or ""
        if not mime_type:
            try:
                mime_type = guess_mime_type(resolved)
            except ValueError as exc:
                return ToolResult.fail("unsupported_file_type", str(exc))

        provider = self._provider or AnthropicDocumentProvider(
            model=ctx.settings.llm_model, max_tokens=ctx.settings.llm_max_tokens,
        )

        try:
            structure = provider.extract_structure(resolved.read_bytes(), mime_type=mime_type)
        except VisionUnavailable as exc:
            return ToolResult.fail("vision_unavailable", str(exc))
        except Exception as exc:  # noqa: BLE001 - a provider failure (auth, network) is one reportable error
            return ToolResult.fail("structure_extraction_failed", f"vision provider failed: {exc}")

        return ToolResult.ok(_structure_data(structure))


ALL_TOOLS = [VisionExtractStructureTool()]


def register_all(registry: ToolRegistry) -> None:
    for tool in ALL_TOOLS:
        registry.register(tool)
