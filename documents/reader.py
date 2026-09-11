"""Read existing DOCX files and extract structured content.

Unlike `documents.docx_writer` which *creates* DOCX files, this reads
an existing one and preserves its structure — headings, paragraphs,
tables — as a `DocumentStructure`-compatible dict.  This enables the
agent to consume DOCX assignments and reports as input, not just
generate them as output.
"""
from __future__ import annotations

from pathlib import Path

from core.errors import SandboxViolation
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.registry import ToolRegistry
from core.tools.result import ToolResult
from core.tools.schema import array, integer, obj, string

_TABLE_SCHEMA = obj({"headers": array(string()), "rows": array(array(string()))})
_SECTION_SCHEMA = obj({
    "heading": string(), "level": integer(), "paragraphs": array(string()),
    "bullets": array(string()), "table": _TABLE_SCHEMA,
})


def _read_docx(path: Path) -> dict:
    """Extract structured content from a DOCX file.

    Returns a dict with 'title', 'sections', 'tables', and
    'paragraphs' (flat body text for sections without headings).
    """
    try:
        import docx
    except ImportError:
        raise FileNotFoundError(
            "the 'python-docx' package is not installed; run `pip install kanna[documents]`"
        )

    doc = docx.Document(str(path))

    # Extract the document title from the first heading at level 0.
    title = ""
    title_para_idx = -1
    for i, para in enumerate(doc.paragraphs):
        if para.style.name == "Title" or (para.style.name.startswith("Heading") and
                                          para.text.strip() and
                                          para.style.name == "Heading 0"):
            title = para.text.strip()
            title_para_idx = i
            break
        # Also accept the first Heading 1 as title if no Heading 0/Title found.
        if para.style.name.startswith("Heading") and para.text.strip():
            try:
                lvl = int(para.style.name.split()[-1])
                if lvl == 0:
                    title = para.text.strip()
                    title_para_idx = i
                    break
            except (IndexError, ValueError):
                pass

    sections = []
    current_section = None

    for i, para in enumerate(doc.paragraphs):
        # Skip the title paragraph.
        if i == title_para_idx:
            continue

        text = para.text.strip()
        if not text:
            continue

        style = para.style.name
        if style.startswith("Heading"):
            # Save previous section if it exists.
            if current_section is not None:
                sections.append(current_section)
            # Extract heading level from style name (e.g. "Heading 1" -> 1).
            try:
                level = int(style.split()[-1])
            except (IndexError, ValueError):
                level = 1
            current_section = {
                "heading": text, "level": level,
                "paragraphs": [], "bullets": [],
            }
        elif "List Bullet" in style:
            if current_section is not None:
                current_section["bullets"].append(text)
            else:
                current_section = {"heading": None, "level": 1,
                                   "paragraphs": [], "bullets": [text]}
        else:
            if current_section is not None:
                current_section["paragraphs"].append(text)
            else:
                current_section = {"heading": None, "level": 1,
                                   "paragraphs": [text], "bullets": []}

    if current_section is not None:
        sections.append(current_section)

    # Extract tables.
    all_tables = []
    for table in doc.tables:
        if not table.rows:
            continue
        headers = [cell.text.strip() for cell in table.rows[0].cells]
        rows = []
        for row in table.rows[1:]:
            rows.append([cell.text.strip() for cell in row.cells])
        all_tables.append({"headers": headers, "rows": rows})

    # Attach standalone tables to the last section or as a new section.
    for tbl in all_tables:
        if sections:
            sections[-1]["table"] = tbl
        else:
            sections.append({"heading": "Tables", "level": 1,
                             "paragraphs": [], "bullets": [], "table": tbl})

    return {
        "title": title,
        "sections": sections,
        "tables": all_tables,
    }


class DocumentReadTool:
    """Reads an existing DOCX file and returns its structured content.

    Preserves headings, paragraphs, bullets, and tables — enabling the
    agent to consume DOCX assignments and reports as input for further
    processing (content authoring, report generation, etc.).
    """

    name = "document_read"
    description = ("Read an existing DOCX file within the sandbox and return its structured "
                   "content: title, sections with headings/paragraphs/bullets, and tables.")
    permission = PermissionLevel.LOW
    input_schema = obj(
        {
            "path": string(description="Path to the DOCX file, within the sandbox"),
        },
        required=("path",),
    )
    output_schema = obj({
        "title": string(),
        "sections": array(_SECTION_SCHEMA),
        "tables": array(_TABLE_SCHEMA),
    })

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            resolved = ctx.sandbox.resolve(args["path"])
        except SandboxViolation as exc:
            return ToolResult.fail("sandbox_violation", str(exc))

        if not resolved.exists() or not resolved.is_file():
            return ToolResult.fail("not_found", f"file does not exist: {resolved}")

        if not resolved.suffix.lower() == ".docx":
            return ToolResult.fail("unsupported_format",
                                   f"only .docx files are supported, got: {resolved.suffix}")

        try:
            data = _read_docx(resolved)
        except FileNotFoundError as exc:
            return ToolResult.fail("document_generation_unavailable", str(exc))
        except Exception as exc:  # noqa: BLE001 — a corrupted file is one reportable error
            return ToolResult.fail("read_failed", f"failed to read DOCX: {exc}")

        return ToolResult.ok(data)


ALL_TOOLS = [DocumentReadTool()]


def register_all(registry: ToolRegistry) -> None:
    for tool in ALL_TOOLS:
        registry.register(tool)
