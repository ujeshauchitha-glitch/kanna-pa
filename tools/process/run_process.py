"""Controlled process execution.

Runs an allowlisted executable as an argv list — never through a shell,
so there is no shell-metacharacter injection surface. The executable name
must be on `allowed_executables`; anything else is rejected before a
process is ever spawned.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from core.errors import SandboxViolation
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.result import ToolResult
from core.tools.schema import array, integer, obj, string

_MAX_OUTPUT_CHARS = 200_000
_DEFAULT_TIMEOUT = 30

# Interpreters/compilers/build tools Kanna is allowed to invoke out of the
# box. Callers that need something else construct their own ProcessTool
# with a different allowlist rather than widening this one.
DEFAULT_ALLOWED_EXECUTABLES = frozenset({
    "python3", "python", "pytest",
    "gcc", "g++", "cc", "c++",
    "rustc", "cargo",
    "javac", "java",
    "node", "npm",
    "octave", "octave-cli",
    "echo", "cat", "ls",
})


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text, False
    return text[:_MAX_OUTPUT_CHARS], True


class ProcessTool:
    name = "process_run"
    description = (
        "Run an allowlisted executable (no shell) with arguments and capture stdout, "
        "stderr, exit code, and execution time."
    )
    permission = PermissionLevel.LOW
    input_schema = obj(
        {
            "command": array(string(), description="argv, e.g. ['python3', 'script.py']"),
            "cwd": string(description="Working directory within the sandbox", default=""),
            "timeout_seconds": integer(description="Kill the process after this many seconds",
                                        minimum=1, maximum=300, default=_DEFAULT_TIMEOUT),
        },
        required=("command",),
    )
    output_schema = obj({
        "stdout": string(), "stderr": string(), "exit_code": integer(),
        "duration_seconds": string(), "timed_out": string(), "truncated": string(),
    })

    def __init__(self, allowed_executables: frozenset[str] = DEFAULT_ALLOWED_EXECUTABLES) -> None:
        self.allowed_executables = allowed_executables

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        command: list[str] = list(args["command"])
        if not command:
            return ToolResult.fail("invalid_command", "command must be a non-empty argv list")

        executable = Path(command[0]).name
        if executable not in self.allowed_executables:
            return ToolResult.fail(
                "executable_not_allowed",
                f"'{executable}' is not in the allowed executables list: {sorted(self.allowed_executables)}",
            )

        cwd_arg = args.get("cwd") or "."
        try:
            cwd = ctx.sandbox.resolve(cwd_arg)
        except SandboxViolation as exc:
            return ToolResult.fail("sandbox_violation", str(exc))
        if not cwd.exists() or not cwd.is_dir():
            return ToolResult.fail("not_found", f"working directory does not exist: {cwd}")

        timeout = int(args.get("timeout_seconds") or _DEFAULT_TIMEOUT)

        start = time.monotonic()
        timed_out = False
        try:
            completed = subprocess.run(
                command, cwd=str(cwd), capture_output=True, text=True,
                timeout=timeout, shell=False,
            )
            stdout, stderr, exit_code = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\n[process killed after exceeding {timeout}s timeout]"
            exit_code = -1
        except OSError as exc:
            return ToolResult.fail("spawn_failed", f"could not start process: {exc}")

        duration = time.monotonic() - start
        stdout, out_truncated = _truncate(stdout)
        stderr, err_truncated = _truncate(stderr)

        return ToolResult.ok({
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "duration_seconds": f"{duration:.3f}",
            "timed_out": str(timed_out),
            "truncated": str(out_truncated or err_truncated),
        })
