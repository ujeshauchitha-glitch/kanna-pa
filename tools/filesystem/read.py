from __future__ import annotations

from core.errors import SandboxViolation
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.result import ToolResult
from core.tools.schema import obj, string, boolean, integer

_MAX_BYTES = 5 * 1024 * 1024  # 5MB — large enough for any assignment/report, small enough to be safe


class ReadFileTool:
    name = "fs_read_file"
    description = "Read the contents of a text file within the sandbox."
    permission = PermissionLevel.LOW
    input_schema = obj(
        {"path": string(description="Path to the file, absolute or relative to the sandbox root")},
        required=("path",),
    )
    output_schema = obj(
        {
            "content": string(),
            "path": string(),
            "size_bytes": integer(),
            "truncated": boolean(),
        },
    )

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            resolved = ctx.sandbox.resolve(args["path"])
        except SandboxViolation as exc:
            return ToolResult.fail("sandbox_violation", str(exc))

        if not resolved.exists():
            return ToolResult.fail("not_found", f"file does not exist: {resolved}")
        if resolved.is_dir():
            return ToolResult.fail("is_a_directory", f"path is a directory, not a file: {resolved}")

        size = resolved.stat().st_size
        raw = resolved.read_bytes()
        truncated = size > _MAX_BYTES
        if truncated:
            raw = raw[:_MAX_BYTES]

        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            return ToolResult.fail("not_text", f"file is not valid UTF-8 text: {resolved}")

        return ToolResult.ok({
            "content": content,
            "path": str(resolved),
            "size_bytes": size,
            "truncated": truncated,
        })
