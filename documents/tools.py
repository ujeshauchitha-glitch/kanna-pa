"""Document generation exposed through the tool registry.

Three tools — one per output format — share an input schema and a base
`execute()` that resolves/checks the destination path the same way
`tools.filesystem.write.WriteFileTool` does (creating a new file is
low-risk; overwriting an existing one is REVIEW-gated, downgraded to
auto-allow for the create case by the same policy rule pattern — see
`core/bootstrap.py::default_policy`). Only the renderer differs.
"""
from __future__ import annotations

from pathlib import Path

from core.errors import DocumentConversionUnavailable, DocumentGenerationUnavailable, SandboxViolation
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.registry import ToolRegistry
from core.tools.result import ToolResult
from core.tools.schema import array, boolean, integer, obj, string
from documents.convert import convert_to_pdf
from documents.docx_writer import render_docx
from documents.model import build_document
from documents.pdf_writer import render_pdf
from documents.pptx_writer import render_pptx

_TABLE_SCHEMA = obj({
    "headers": array(string(), description="Column headers"),
    "rows": array(array(string()), description="Table rows, each a list of cell values"),
}, required=("headers",))

_SECTION_SCHEMA = obj({
    "heading": string(default="", description="Section heading (or slide title for PPTX)"),
    "level": integer(default=1, minimum=1, maximum=9,
                      description="Heading depth for DOCX/PDF; ignored for PPTX (one slide per section)"),
    "paragraphs": array(string(), description="Body paragraphs"),
    "bullets": array(string(), description="Bullet points (preferred over paragraphs for PPTX)"),
    "table": _TABLE_SCHEMA,
})

DOCUMENT_INPUT_SCHEMA = obj(
    {
        "title": string(description="Document title"),
        "subtitle": string(default=""),
        "author": string(default=""),
        "path": string(description="Destination path within the sandbox"),
        "sections": array(_SECTION_SCHEMA, description="Ordered content sections"),
        "overwrite": boolean(description="Allow overwriting an existing file", default=False),
    },
    required=("title", "path"),
)

_OUTPUT_SCHEMA = obj({"path": string(), "bytes_written": integer()})


class _DocumentGenerateTool:
    """Base for the three format-specific generators — only `_renderer` differs."""

    permission = PermissionLevel.REVIEW
    input_schema = DOCUMENT_INPUT_SCHEMA
    output_schema = _OUTPUT_SCHEMA

    @staticmethod
    def _renderer(document, path: Path) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            resolved = ctx.sandbox.resolve(args["path"])
        except SandboxViolation as exc:
            return ToolResult.fail("sandbox_violation", str(exc))

        overwrite = bool(args.get("overwrite", False))
        existed = resolved.exists()
        if existed and resolved.is_dir():
            return ToolResult.fail("is_a_directory", f"path is a directory: {resolved}")
        if existed and not overwrite:
            return ToolResult.fail(
                "already_exists", f"'{resolved}' already exists; pass overwrite=true to replace it"
            )

        document = build_document(args)
        try:
            self._renderer(document, resolved)
        except DocumentGenerationUnavailable as exc:
            return ToolResult.fail("document_generation_unavailable", str(exc))

        result = ToolResult.ok({"path": str(resolved), "bytes_written": resolved.stat().st_size})
        if existed:
            result.files_modified = [str(resolved)]
        else:
            result.files_created = [str(resolved)]
        return result


class DocumentGenerateDocxTool(_DocumentGenerateTool):
    name = "document_generate_docx"
    description = "Generate a Word document (.docx) from structured title/sections content."
    _renderer = staticmethod(render_docx)


class DocumentGeneratePptxTool(_DocumentGenerateTool):
    name = "document_generate_pptx"
    description = ("Generate a PowerPoint deck (.pptx) from structured content — one slide per "
                    "section, using bullets (preferred) or paragraphs as body text.")
    _renderer = staticmethod(render_pptx)


class DocumentGeneratePdfTool(_DocumentGenerateTool):
    name = "document_generate_pdf"
    description = "Generate a PDF (.pdf) from structured title/sections content."
    _renderer = staticmethod(render_pdf)


_CONVERT_INPUT_SCHEMA = obj(
    {
        "source_path": string(description="Path to an existing DOCX/PPTX file, within the sandbox"),
        "dest_path": string(description="Destination PDF path, within the sandbox"),
        "overwrite": boolean(description="Allow overwriting an existing file", default=False),
    },
    required=("source_path", "dest_path"),
)


class DocumentConvertToPdfTool:
    """Converts an existing DOCX/PPTX file to PDF via LibreOffice's headless
    mode (`documents.convert.convert_to_pdf`) — see `docs/DOCUMENTS.md`.
    Sandboxed and overwrite-gated the same way as the generate-* tools.
    """

    name = "document_convert_to_pdf"
    description = "Convert an existing DOCX or PPTX file (within the sandbox) to PDF."
    permission = PermissionLevel.REVIEW
    input_schema = _CONVERT_INPUT_SCHEMA
    output_schema = _OUTPUT_SCHEMA

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            source = ctx.sandbox.resolve(args["source_path"])
            dest = ctx.sandbox.resolve(args["dest_path"])
        except SandboxViolation as exc:
            return ToolResult.fail("sandbox_violation", str(exc))

        if not source.exists() or not source.is_file():
            return ToolResult.fail("not_found", f"source file does not exist: {source}")

        overwrite = bool(args.get("overwrite", False))
        dest_existed = dest.exists()
        if dest_existed and dest.is_dir():
            return ToolResult.fail("is_a_directory", f"destination is a directory: {dest}")
        if dest_existed and not overwrite:
            return ToolResult.fail(
                "already_exists", f"'{dest}' already exists; pass overwrite=true to replace it"
            )

        try:
            convert_to_pdf(source, dest)
        except DocumentConversionUnavailable as exc:
            return ToolResult.fail("document_conversion_unavailable", str(exc))

        result = ToolResult.ok({"path": str(dest), "bytes_written": dest.stat().st_size})
        if dest_existed:
            result.files_modified = [str(dest)]
        else:
            result.files_created = [str(dest)]
        return result


ALL_TOOLS = [
    DocumentGenerateDocxTool(), DocumentGeneratePptxTool(), DocumentGeneratePdfTool(),
    DocumentConvertToPdfTool(),
]


def register_all(registry: ToolRegistry) -> None:
    for tool in ALL_TOOLS:
        registry.register(tool)
