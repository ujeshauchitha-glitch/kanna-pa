from __future__ import annotations

from core.errors import SandboxViolation
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.result import ToolResult
from core.tools.schema import array, boolean, integer, obj, string


class ListDirectoryTool:
    name = "fs_list_directory"
    description = "List the entries of a directory within the sandbox."
    permission = PermissionLevel.LOW
    input_schema = obj(
        {"path": string(description="Directory path"),
         "recursive": boolean(description="Recurse into subdirectories", default=False)},
        required=("path",),
    )
    output_schema = obj({
        "path": string(),
        "entries": array(obj({
            "name": string(), "path": string(), "is_dir": boolean(), "size_bytes": integer(),
        })),
    })

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            resolved = ctx.sandbox.resolve(args["path"])
        except SandboxViolation as exc:
            return ToolResult.fail("sandbox_violation", str(exc))

        if not resolved.exists():
            return ToolResult.fail("not_found", f"directory does not exist: {resolved}")
        if not resolved.is_dir():
            return ToolResult.fail("not_a_directory", f"path is not a directory: {resolved}")

        recursive = bool(args.get("recursive", False))
        iterator = resolved.rglob("*") if recursive else resolved.iterdir()

        entries = []
        for entry in sorted(iterator, key=lambda p: str(p)):
            try:
                size = entry.stat().st_size if entry.is_file() else 0
            except OSError:
                size = 0
            entries.append({
                "name": entry.name,
                "path": str(entry),
                "is_dir": entry.is_dir(),
                "size_bytes": size,
            })

        return ToolResult.ok({"path": str(resolved), "entries": entries})
