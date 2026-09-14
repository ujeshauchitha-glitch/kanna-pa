from __future__ import annotations

import shutil

from core.errors import SandboxViolation
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.result import ToolResult
from core.tools.schema import boolean, obj, string


class DeleteFileTool:
    """Deletes a file or (with `recursive=true`) a directory tree. Always REVIEW-level — irreversible."""

    name = "fs_delete"
    description = "Delete a file or directory within the sandbox. Irreversible — always requires approval."
    permission = PermissionLevel.REVIEW
    input_schema = obj(
        {
            "path": string(description="Path to delete"),
            "recursive": boolean(description="Required to delete a non-empty directory", default=False),
        },
        required=("path",),
    )
    output_schema = obj({"path": string()})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            resolved = ctx.sandbox.resolve(args["path"])
        except SandboxViolation as exc:
            return ToolResult.fail("sandbox_violation", str(exc))

        if not resolved.exists():
            return ToolResult.fail("not_found", f"path does not exist: {resolved}")

        if resolved.is_dir():
            if not args.get("recursive", False):
                return ToolResult.fail("is_a_directory", "pass recursive=true to delete a directory")
            shutil.rmtree(resolved)
        else:
            resolved.unlink()

        result = ToolResult.ok({"path": str(resolved)})
        result.files_deleted = [str(resolved)]
        return result
