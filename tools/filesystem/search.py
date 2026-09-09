from __future__ import annotations

import re

from core.errors import SandboxViolation
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.result import ToolResult
from core.tools.schema import array, boolean, integer, obj, string

_MAX_MATCHES = 500
_MAX_FILE_BYTES = 2 * 1024 * 1024


class SearchFilesTool:
    name = "fs_search_files"
    description = "Search for a regular expression across text files under a directory in the sandbox."
    permission = PermissionLevel.LOW
    input_schema = obj(
        {
            "path": string(description="Directory to search under"),
            "pattern": string(description="Regular expression to search for"),
            "glob": string(description="Filename glob filter, e.g. '*.py'", default="*"),
        },
        required=("path", "pattern"),
    )
    output_schema = obj({
        "matches": array(obj({
            "path": string(), "line_number": integer(), "line": string(),
        })),
        "truncated": boolean(),
    })

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            resolved = ctx.sandbox.resolve(args["path"])
        except SandboxViolation as exc:
            return ToolResult.fail("sandbox_violation", str(exc))

        if not resolved.exists() or not resolved.is_dir():
            return ToolResult.fail("not_found", f"directory does not exist: {resolved}")

        try:
            regex = re.compile(args["pattern"])
        except re.error as exc:
            return ToolResult.fail("invalid_pattern", f"invalid regular expression: {exc}")

        glob_pattern = args.get("glob") or "*"
        matches = []
        truncated = False
        for file_path in sorted(resolved.rglob(glob_pattern)):
            if not file_path.is_file():
                continue
            try:
                if file_path.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append({"path": str(file_path), "line_number": i, "line": line[:500]})
                    if len(matches) >= _MAX_MATCHES:
                        truncated = True
                        break
            if truncated:
                break

        return ToolResult.ok({"matches": matches, "truncated": truncated})
