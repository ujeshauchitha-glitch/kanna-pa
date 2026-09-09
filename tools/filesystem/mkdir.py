from __future__ import annotations

from core.errors import SandboxViolation
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.result import ToolResult
from core.tools.schema import obj, string


class CreateDirectoryTool:
    name = "fs_create_directory"
    description = "Create a directory (and parents) within the sandbox."
    permission = PermissionLevel.LOW
    input_schema = obj({"path": string(description="Directory to create")}, required=("path",))
    output_schema = obj({"path": string(), "created": string()})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            resolved = ctx.sandbox.resolve(args["path"])
        except SandboxViolation as exc:
            return ToolResult.fail("sandbox_violation", str(exc))

        if resolved.exists() and not resolved.is_dir():
            return ToolResult.fail("not_a_directory", f"a non-directory already exists at {resolved}")

        already_existed = resolved.exists()
        resolved.mkdir(parents=True, exist_ok=True)

        result = ToolResult.ok({"path": str(resolved), "created": str(not already_existed)})
        if not already_existed:
            result.files_created = [str(resolved)]
        return result
