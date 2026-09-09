from __future__ import annotations

import pytest

pytest.importorskip("docx", reason="python-docx not installed (kanna[documents] extra)")
pytest.importorskip("pptx", reason="python-pptx not installed (kanna[documents] extra)")
pytest.importorskip("reportlab", reason="reportlab not installed (kanna[documents] extra)")

from documents.tools import (  # noqa: E402
    DocumentGenerateDocxTool, DocumentGeneratePdfTool, DocumentGeneratePptxTool,
)

_ARGS = {
    "title": "Report", "path": "out.docx",
    "sections": [{"heading": "Intro", "paragraphs": ["hello"]}],
}


@pytest.mark.parametrize("tool_cls,ext", [
    (DocumentGenerateDocxTool, "docx"), (DocumentGeneratePptxTool, "pptx"),
    (DocumentGeneratePdfTool, "pdf"),
])
def test_generate_creates_file(ctx, tmp_path, tool_cls, ext):
    tool = tool_cls()
    args = {**_ARGS, "path": f"out.{ext}"}
    result = tool.execute(args, ctx)
    assert result.success
    assert result.data["bytes_written"] > 0
    assert result.files_created == [str(tmp_path / f"out.{ext}")]
    assert (tmp_path / f"out.{ext}").exists()


def test_refuses_overwrite_without_flag(ctx):
    tool = DocumentGenerateDocxTool()
    tool.execute(_ARGS, ctx)
    result = tool.execute(_ARGS, ctx)
    assert not result.success
    assert result.error.code == "already_exists"


def test_overwrite_with_flag_marks_modified(ctx, tmp_path):
    tool = DocumentGenerateDocxTool()
    tool.execute(_ARGS, ctx)
    result = tool.execute({**_ARGS, "overwrite": True}, ctx)
    assert result.success
    assert result.files_modified == [str(tmp_path / "out.docx")]


def test_rejects_sandbox_escape(ctx):
    tool = DocumentGenerateDocxTool()
    result = tool.execute({**_ARGS, "path": "../../etc/report.docx"}, ctx)
    assert not result.success
    assert result.error.code == "sandbox_violation"


def test_rejects_path_that_is_a_directory(ctx, tmp_path):
    (tmp_path / "adir").mkdir()
    tool = DocumentGenerateDocxTool()
    result = tool.execute({**_ARGS, "path": "adir"}, ctx)
    assert not result.success
    assert result.error.code == "is_a_directory"


def test_generate_with_table_and_bullets(ctx, tmp_path):
    tool = DocumentGeneratePptxTool()
    args = {
        "title": "Deck", "path": "deck.pptx",
        "sections": [
            {"heading": "Findings", "bullets": ["a", "b"]},
            {"heading": "Numbers", "table": {"headers": ["X", "Y"], "rows": [["1", "2"]]}},
        ],
    }
    result = tool.execute(args, ctx)
    assert result.success


def test_registered_at_review_permission():
    assert DocumentGenerateDocxTool().permission.name == "REVIEW"
    assert DocumentGeneratePptxTool().permission.name == "REVIEW"
    assert DocumentGeneratePdfTool().permission.name == "REVIEW"
