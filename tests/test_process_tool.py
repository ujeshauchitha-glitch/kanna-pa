from __future__ import annotations

import sys
from pathlib import Path
from tools.process.run_process import ProcessTool


def test_run_python_captures_stdout_and_exit_code(ctx):
    tool = ProcessTool(allowed_executables=frozenset({Path(sys.executable).name}))
    result = tool.execute({"command": [sys.executable, "-c", "print('hello')"]}, ctx)
    assert result.success
    assert result.data["stdout"].strip() == "hello"
    assert result.data["exit_code"] == 0


def test_run_captures_stderr_and_nonzero_exit(ctx):
    tool = ProcessTool(allowed_executables=frozenset({Path(sys.executable).name}))
    result = tool.execute(
        {"command": [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"]}, ctx
    )
    assert result.success  # the tool call succeeded even though the *process* exited nonzero
    assert result.data["exit_code"] == 3
    assert "boom" in result.data["stderr"]


def test_disallowed_executable_is_rejected(ctx):
    tool = ProcessTool(allowed_executables=frozenset({Path(sys.executable).name}))
    result = tool.execute({"command": ["rm", "-rf", "/"]}, ctx)
    assert not result.success
    assert result.error.code == "executable_not_allowed"


def test_empty_command_rejected(ctx):
    tool = ProcessTool(allowed_executables=frozenset({Path(sys.executable).name}))
    result = tool.execute({"command": []}, ctx)
    assert not result.success
    assert result.error.code == "invalid_command"


def test_timeout_kills_process(ctx):
    tool = ProcessTool(allowed_executables=frozenset({Path(sys.executable).name}))
    result = tool.execute(
        {"command": [sys.executable, "-c", "import time; time.sleep(5)"], "timeout_seconds": 1}, ctx
    )
    assert result.success
    assert result.data["timed_out"] == "True"


def test_output_truncation():
    from tools.process.run_process import _truncate, _MAX_OUTPUT_CHARS
    text = "x" * (_MAX_OUTPUT_CHARS + 100)
    truncated, was_truncated = _truncate(text)
    assert was_truncated
    assert len(truncated) == _MAX_OUTPUT_CHARS


def test_timeout_preserves_partial_output(ctx, monkeypatch):
    import subprocess
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, output=b"partial", stderr=b"error")
    monkeypatch.setattr(subprocess, "run", timeout)
    tool = ProcessTool(allowed_executables=frozenset({Path(sys.executable).name}))
    result = tool.execute({"command": [sys.executable, "-u", "-c",
        "import sys,time; print('partial'); sys.stderr.write('error'); sys.stderr.flush(); time.sleep(5)"],
        "timeout_seconds": 1}, ctx)
    assert result.success
    assert "partial" in result.data["stdout"]
    assert "error" in result.data["stderr"]
    assert result.data["timed_out"] == "True"


def test_custom_allowlist_restricts_further(ctx):
    tool = ProcessTool(allowed_executables=frozenset({"echo"}))
    result = tool.execute({"command": [sys.executable, "-c", "print(1)"]}, ctx)
    assert not result.success
    assert result.error.code == "executable_not_allowed"


def test_default_allowlist_includes_make():
    # 'make' exists specifically so a tools.scaffold-generated C/C++
    # project's Makefile is actually runnable through process_run, not
    # just generated — see tests/test_scaffold_build.py for the
    # end-to-end proof.
    from tools.process.run_process import DEFAULT_ALLOWED_EXECUTABLES
    assert "make" in DEFAULT_ALLOWED_EXECUTABLES
