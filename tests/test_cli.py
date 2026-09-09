from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str], tmp_path: Path, cwd: Path | None = None) -> subprocess.CompletedProcess:
    import os

    env = dict(**os.environ)
    env["KANNA_HOME"] = str(tmp_path / "kanna_home")
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "main.py"), *args],
        cwd=str(cwd or tmp_path), env=env, capture_output=True, text=True, timeout=30,
    )


def test_init(tmp_path):
    result = _run(["init"], tmp_path)
    assert result.returncode == 0
    assert "Kanna initialized" in result.stdout


def test_tools_list(tmp_path):
    result = _run(["tools", "list"], tmp_path)
    assert result.returncode == 0
    assert "fs_read_file" in result.stdout
    assert "finance_add_transaction" in result.stdout


def test_db_migrate(tmp_path):
    result = _run(["db", "migrate"], tmp_path)
    assert result.returncode == 0


def test_finance_add_and_query(tmp_path):
    add_result = _run(["finance", "add", "I spent 340 on lunch"], tmp_path)
    assert add_result.returncode == 0
    assert "340.00" in add_result.stdout

    query_result = _run(["finance", "query", "How much did I spend on food this month?"], tmp_path)
    assert query_result.returncode == 0
    assert "340.00" in query_result.stdout


def test_ask_list_directory(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("hello")

    result = _run(["ask", "list the files in ./docs"], tmp_path)
    assert result.returncode == 0


def test_task_lifecycle(tmp_path):
    add_result = _run(["task", "add", "Write report"], tmp_path)
    assert add_result.returncode == 0
    task_id = add_result.stdout.split()[2].rstrip(":")

    list_result = _run(["task", "list"], tmp_path)
    assert task_id in list_result.stdout

    complete_result = _run(["task", "complete", task_id], tmp_path)
    assert complete_result.returncode == 0
