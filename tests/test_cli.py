from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

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
    assert "document_generate_docx" in result.stdout


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


def test_document_generate(tmp_path):
    import json

    content = {
        "title": "Report", "path": "report.pdf",
        "sections": [{"heading": "Intro", "paragraphs": ["hello"]}],
    }
    content_path = tmp_path / "content.json"
    content_path.write_text(json.dumps(content))

    result = _run(["document", "generate", str(content_path), "--format", "pdf"], tmp_path)
    assert result.returncode == 0
    assert (tmp_path / "report.pdf").exists()

    # Second run without --overwrite fails cleanly.
    second = _run(["document", "generate", str(content_path), "--format", "pdf"], tmp_path)
    assert second.returncode == 1
    assert "already exists" in second.stderr

    third = _run(["document", "generate", str(content_path), "--format", "pdf", "--overwrite"],
                  tmp_path)
    assert third.returncode == 0


def test_trust_lifecycle(tmp_path):
    list_result = _run(["trust", "list"], tmp_path)
    assert list_result.returncode == 0
    assert "No standing approval rules" in list_result.stdout

    add_result = _run(
        ["trust", "add", "computer_click", "--arg", "x=5", "--arg", "y=5", "--note", "safe corner"],
        tmp_path,
    )
    assert add_result.returncode == 0
    assert "computer_click" in add_result.stdout

    list_result = _run(["trust", "list"], tmp_path)
    assert list_result.returncode == 0
    assert "computer_click" in list_result.stdout
    assert "safe corner" in list_result.stdout
    rule_id = list_result.stdout.split("]")[0].lstrip("[")

    remove_result = _run(["trust", "remove", rule_id], tmp_path)
    assert remove_result.returncode == 0

    list_result = _run(["trust", "list"], tmp_path)
    assert "No standing approval rules" in list_result.stdout


def test_trust_add_rejects_unknown_tool(tmp_path):
    result = _run(["trust", "add", "not_a_real_tool"], tmp_path)
    assert result.returncode == 1
    assert "no tool registered" in result.stderr


def test_scheduler_add_list_and_tick(tmp_path):
    add_result = _run(
        ["scheduler", "add", "daily check-in", "list the files in ./docs",
         "--kind", "once", "--run-at", "2000-01-01T00:00:00"],  # already due whenever this runs
        tmp_path,
    )
    assert add_result.returncode == 0
    assert "daily check-in" in add_result.stdout

    list_result = _run(["scheduler", "list"], tmp_path)
    assert list_result.returncode == 0
    assert "daily check-in" in list_result.stdout
    line = next(ln for ln in list_result.stdout.splitlines() if "daily check-in" in ln)
    schedule_id = line.split("]")[1].split()[0]  # "[active  ] <id>  <name>  ..."

    tick_result = _run(["scheduler", "tick"], tmp_path)
    assert tick_result.returncode == 0
    assert "daily check-in" in tick_result.stdout

    remove_result = _run(["scheduler", "remove", schedule_id], tmp_path)
    assert remove_result.returncode == 0
    assert "Deactivated" in remove_result.stdout


def test_scheduler_add_rejects_missing_required_fields(tmp_path):
    result = _run(["scheduler", "add", "bad", "do something", "--kind", "once"], tmp_path)
    assert result.returncode == 1
    assert "--run-at" in result.stderr


def test_scheduler_daemon_bounded_ticks(tmp_path):
    _run(["scheduler", "add", "one-off", "list the files in ./docs",
          "--kind", "once", "--run-at", "2000-01-01T00:00:00"], tmp_path)

    result = _run(["scheduler", "daemon", "--ticks", "2", "--interval-seconds", "0"], tmp_path)
    assert result.returncode == 0
    assert "one-off" in result.stdout


def test_voice_fails_cleanly_without_audio_capability(tmp_path):
    """On a machine missing a voice dependency (numpy/sounddevice/
    SpeechRecognition not installed) or the PortAudio shared library
    (this sandbox, most CI runners), `kanna voice` must fail the same
    clean, typed way every other optional-capability tool does — a
    message and exit code 1, never a raw traceback. This test only
    asserts something when one of those is actually true here.
    """
    try:
        import sounddevice
        sounddevice.query_devices()
    except (ImportError, OSError):
        pass
    else:
        pytest.skip("voice deps + PortAudio both available on this machine; nothing to assert here")

    result = _run(["voice"], tmp_path)
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "error:" in result.stderr


def test_task_lifecycle(tmp_path):
    add_result = _run(["task", "add", "Write report"], tmp_path)
    assert add_result.returncode == 0
    task_id = add_result.stdout.split()[2].rstrip(":")

    list_result = _run(["task", "list"], tmp_path)
    assert task_id in list_result.stdout

    complete_result = _run(["task", "complete", task_id], tmp_path)
    assert complete_result.returncode == 0
