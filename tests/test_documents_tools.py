from __future__ import annotations

import pytest

pytest.importorskip("docx", reason="python-docx not installed (kanna[documents] extra)")
pytest.importorskip("pptx", reason="python-pptx not installed (kanna[documents] extra)")
pytest.importorskip("reportlab", reason="reportlab not installed (kanna[documents] extra)")

from documents.tools import (  # noqa: E402
    DocumentConvertToPdfTool, DocumentGenerateDocxTool, DocumentGeneratePdfTool,
    DocumentGeneratePptxTool,
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


# -- DocumentConvertToPdfTool — monkeypatched convert_to_pdf, no LibreOffice needed --

def _fake_convert(source, dest, **kwargs):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"%PDF-fake")


def test_convert_tool_creates_pdf(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr("documents.tools.convert_to_pdf", _fake_convert)
    (tmp_path / "in.docx").write_bytes(b"fake docx")

    tool = DocumentConvertToPdfTool()
    result = tool.execute({"source_path": "in.docx", "dest_path": "out.pdf"}, ctx)

    assert result.success
    assert result.data["path"] == str(tmp_path / "out.pdf")
    assert result.files_created == [str(tmp_path / "out.pdf")]
    assert (tmp_path / "out.pdf").read_bytes() == b"%PDF-fake"


def test_convert_tool_rejects_missing_source(ctx):
    tool = DocumentConvertToPdfTool()
    result = tool.execute({"source_path": "nope.docx", "dest_path": "out.pdf"}, ctx)
    assert not result.success
    assert result.error.code == "not_found"


def test_convert_tool_rejects_sandbox_escape_on_source(ctx):
    tool = DocumentConvertToPdfTool()
    result = tool.execute({"source_path": "../../etc/passwd", "dest_path": "out.pdf"}, ctx)
    assert not result.success
    assert result.error.code == "sandbox_violation"


def test_convert_tool_rejects_sandbox_escape_on_dest(ctx, tmp_path):
    (tmp_path / "in.docx").write_bytes(b"fake docx")
    tool = DocumentConvertToPdfTool()
    result = tool.execute({"source_path": "in.docx", "dest_path": "../../etc/out.pdf"}, ctx)
    assert not result.success
    assert result.error.code == "sandbox_violation"


def test_convert_tool_refuses_overwrite_without_flag(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr("documents.tools.convert_to_pdf", _fake_convert)
    (tmp_path / "in.docx").write_bytes(b"fake docx")
    (tmp_path / "out.pdf").write_bytes(b"already here")

    tool = DocumentConvertToPdfTool()
    result = tool.execute({"source_path": "in.docx", "dest_path": "out.pdf"}, ctx)
    assert not result.success
    assert result.error.code == "already_exists"


def test_convert_tool_overwrite_with_flag_marks_modified(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr("documents.tools.convert_to_pdf", _fake_convert)
    (tmp_path / "in.docx").write_bytes(b"fake docx")
    (tmp_path / "out.pdf").write_bytes(b"already here")

    tool = DocumentConvertToPdfTool()
    result = tool.execute(
        {"source_path": "in.docx", "dest_path": "out.pdf", "overwrite": True}, ctx
    )
    assert result.success
    assert result.files_modified == [str(tmp_path / "out.pdf")]


def test_convert_tool_reports_conversion_unavailable(ctx, tmp_path, monkeypatch):
    from core.errors import DocumentConversionUnavailable

    def _raising_convert(source, dest, **kwargs):
        raise DocumentConversionUnavailable("no LibreOffice binary found")

    monkeypatch.setattr("documents.tools.convert_to_pdf", _raising_convert)
    (tmp_path / "in.docx").write_bytes(b"fake docx")

    tool = DocumentConvertToPdfTool()
    result = tool.execute({"source_path": "in.docx", "dest_path": "out.pdf"}, ctx)
    assert not result.success
    assert result.error.code == "document_conversion_unavailable"


def test_convert_tool_registered_at_review_permission():
    assert DocumentConvertToPdfTool().permission.name == "REVIEW"
