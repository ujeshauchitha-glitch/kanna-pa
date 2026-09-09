"""Builds and runs every generated project skeleton for real, using the
same compilers `tools.process.run_process.ProcessTool` allowlists
(`gcc`/`g++`/`javac`/`java`) — not just asserting the generated text
looks plausible. Skipped per-language if that language's compiler isn't
installed, the same `shutil.which`-gated pattern `tests/
test_fedora_agent.py` uses for `xdotool`/`scrot`/`xclip`.

`tests/test_scaffold_tools.py` covers the tool layer (sandboxing,
overwrite handling, unsupported language) without needing any compiler
at all.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from tools.process.run_process import ProcessTool
from tools.scaffold.templates import c_project, cpp_project, java_project
from tools.scaffold.tools import ProjectScaffoldTool


def _write_project(tmp_path, files) -> None:
    for f in files:
        path = tmp_path / f.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f.content, encoding="utf-8")


@pytest.mark.skipif(shutil.which("gcc") is None or shutil.which("make") is None,
                     reason="gcc/make not installed")
def test_c_project_builds_and_runs(tmp_path):
    _write_project(tmp_path, c_project("democ"))

    build = subprocess.run(["make"], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert build.returncode == 0, build.stderr

    run = subprocess.run(["make", "run"], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert "Hello from democ!" in run.stdout


@pytest.mark.skipif(shutil.which("g++") is None or shutil.which("make") is None,
                     reason="g++/make not installed")
def test_cpp_project_builds_and_runs(tmp_path):
    _write_project(tmp_path, cpp_project("democpp"))

    build = subprocess.run(["make"], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert build.returncode == 0, build.stderr

    run = subprocess.run(["make", "run"], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert "Hello from democpp!" in run.stdout


@pytest.mark.skipif(shutil.which("javac") is None or shutil.which("java") is None,
                     reason="javac/java not installed")
def test_java_project_builds_and_runs(tmp_path):
    _write_project(tmp_path, java_project("demojava"))

    build = subprocess.run(["javac", "-d", "build", "src/Main.java"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert build.returncode == 0, build.stderr

    run = subprocess.run(["java", "-cp", "build", "Main"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert "Hello from demojava!" in run.stdout


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
def test_c_makefile_clean_target_removes_binary(tmp_path):
    _write_project(tmp_path, c_project("cleanme"))
    subprocess.run(["make"], cwd=tmp_path, capture_output=True, text=True, timeout=30, check=True)
    assert (tmp_path / "cleanme").exists()

    subprocess.run(["make", "clean"], cwd=tmp_path, capture_output=True, text=True, timeout=30,
                    check=True)
    assert not (tmp_path / "cleanme").exists()


@pytest.mark.skipif(shutil.which("gcc") is None or shutil.which("make") is None,
                     reason="gcc/make not installed")
def test_scaffolded_c_project_is_buildable_through_process_run(ctx, tmp_path):
    """End-to-end proof that what `project_scaffold` generates and what
    `process_run` can actually execute agree with each other — 'make' was
    added to `ProcessTool.DEFAULT_ALLOWED_EXECUTABLES` specifically so
    this works, not just direct compiler invocation."""
    ProjectScaffoldTool().execute({"language": "c", "path": "integ"}, ctx)

    process = ProcessTool()
    build = process.execute({"command": ["make"], "cwd": "integ"}, ctx)
    assert build.success, build.data.get("stderr")
    assert build.data["exit_code"] == 0

    run = process.execute({"command": ["make", "run"], "cwd": "integ"}, ctx)
    assert run.success
    assert run.data["exit_code"] == 0
    assert "Hello from integ!" in run.data["stdout"]
