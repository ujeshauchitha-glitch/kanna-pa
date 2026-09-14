from __future__ import annotations

from core.errors import SandboxViolation
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.result import ToolResult
from core.tools.schema import boolean, integer, obj, string


class FileInfoTool:
    name = "fs_file_info"
    description = "Inspect a path within the sandbox: existence, type, and size, without reading its contents."
    permission = PermissionLevel.LOW
    input_schema = obj({"path": string()}, required=("path",))
    output_schema = obj({
        "path": string(), "exists": boolean(), "is_file": boolean(), "is_dir": boolean(),
        "size_bytes": integer(), "modified_at": string(),
    })

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            resolved = ctx.sandbox.resolve(args["path"])
        except SandboxViolation as exc:
            return ToolResult.fail("sandbox_violation", str(exc))

        if not resolved.exists():
            return ToolResult.ok({
                "path": str(resolved), "exists": False, "is_file": False, "is_dir": False,
                "size_bytes": 0, "modified_at": "",
            })

        stat = resolved.stat()
        from datetime import datetime, timezone
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

        return ToolResult.ok({
            "path": str(resolved),
            "exists": True,
            "is_file": resolved.is_file(),
            "is_dir": resolved.is_dir(),
            "size_bytes": stat.st_size if resolved.is_file() else 0,
            "modified_at": modified_at,
        })
