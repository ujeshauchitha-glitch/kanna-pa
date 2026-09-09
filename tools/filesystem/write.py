from __future__ import annotations

from core.errors import SandboxViolation
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.result import ToolResult
from core.tools.schema import boolean, integer, obj, string


class WriteFileTool:
    """Creates a new file, or overwrites an existing one only with `overwrite=true`.

    Overwriting an existing file is REVIEW-level (irreversible without a
    backup) — creating a brand-new file is LOW-level. We can't know which
    case applies until `execute()` checks the filesystem, so the tool is
    registered at REVIEW and self-downgrades the *result* is not possible;
    instead the policy predicate for this tool (see permissions setup)
    inspects `overwrite` in args. See `interfaces/cli/app.py` wiring.
    """

    name = "fs_write_file"
    description = "Write text content to a file. Creating a new file is safe; overwriting an existing one requires approval."
    permission = PermissionLevel.REVIEW
    input_schema = obj(
        {
            "path": string(description="Destination path"),
            "content": string(description="Text content to write"),
            "overwrite": boolean(description="Allow overwriting an existing file", default=False),
        },
        required=("path", "content"),
    )
    output_schema = obj({"path": string(), "bytes_written": integer()})

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
                "already_exists",
                f"'{resolved}' already exists; pass overwrite=true to replace it",
            )

        resolved.parent.mkdir(parents=True, exist_ok=True)
        content: str = args["content"]
        resolved.write_text(content, encoding="utf-8")

        result = ToolResult.ok({"path": str(resolved), "bytes_written": len(content.encode("utf-8"))})
        if existed:
            result.files_modified = [str(resolved)]
        else:
            result.files_created = [str(resolved)]
        return result
