from __future__ import annotations

from core.errors import VisionUnavailable
from vision.document.base import DocumentStructure, StructureSection, StructureTable
from vision.document.fake import FakeDocumentProvider
from vision.tools import VisionExtractStructureTool


def test_tool_reads_sandboxed_file_and_returns_structure(ctx, tmp_path):
    (tmp_path / "assignment.pdf").write_bytes(b"%PDF fake bytes")
    provider = FakeDocumentProvider(structure_extractions=[DocumentStructure(
        title="Assignment 3",
        sections=[StructureSection(heading="Question 1", paragraphs=["Explain X."])],
    )])
    tool = VisionExtractStructureTool(provider=provider)

    result = tool.execute({"path": "assignment.pdf"}, ctx)

    assert result.success
    assert result.data["title"] == "Assignment 3"
    assert result.data["sections"] == [
        {"heading": "Question 1", "level": 1, "paragraphs": ["Explain X."], "bullets": []},
    ]
    assert provider.structure_calls[0][1] == "application/pdf"


def test_tool_includes_table_key_only_when_a_table_is_present(ctx, tmp_path):
    (tmp_path / "report.pdf").write_bytes(b"%PDF fake")
    provider = FakeDocumentProvider(structure_extractions=[DocumentStructure(
        sections=[StructureSection(heading="Figures",
                                    table=StructureTable(headers=["Q1"], rows=[["100"]]))],
    )])
    tool = VisionExtractStructureTool(provider=provider)

    result = tool.execute({"path": "report.pdf"}, ctx)

    assert result.success
    assert result.data["sections"][0]["table"] == {"headers": ["Q1"], "rows": [["100"]]}


def test_tool_rejects_missing_file(ctx):
    tool = VisionExtractStructureTool(provider=FakeDocumentProvider())
    result = tool.execute({"path": "does-not-exist.pdf"}, ctx)
    assert not result.success
    assert result.error.code == "not_found"


def test_tool_rejects_sandbox_escape(ctx):
    tool = VisionExtractStructureTool(provider=FakeDocumentProvider())
    result = tool.execute({"path": "../../etc/passwd"}, ctx)
    assert not result.success
    assert result.error.code == "sandbox_violation"


def test_tool_rejects_unsupported_extension(ctx, tmp_path):
    (tmp_path / "notes.txt").write_text("not an image")
    tool = VisionExtractStructureTool(provider=FakeDocumentProvider())
    result = tool.execute({"path": "notes.txt"}, ctx)
    assert not result.success
    assert result.error.code == "unsupported_file_type"


def test_tool_honors_explicit_mime_type_override(ctx, tmp_path):
    (tmp_path / "doc.bin").write_bytes(b"fake pdf bytes")
    provider = FakeDocumentProvider(structure_extractions=[DocumentStructure(title="x")])
    tool = VisionExtractStructureTool(provider=provider)
    result = tool.execute({"path": "doc.bin", "mime_type": "application/pdf"}, ctx)
    assert result.success
    assert provider.structure_calls[0][1] == "application/pdf"


def test_tool_reports_vision_unavailable(ctx, tmp_path):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF fake")

    class _UnavailableProvider(FakeDocumentProvider):
        def extract_structure(self, file_bytes, *, mime_type):
            raise VisionUnavailable("ANTHROPIC_API_KEY is not set")

    tool = VisionExtractStructureTool(provider=_UnavailableProvider())
    result = tool.execute({"path": "doc.pdf"}, ctx)
    assert not result.success
    assert result.error.code == "vision_unavailable"


def test_tool_reports_generic_provider_failure(ctx, tmp_path):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF fake")

    class _FailingProvider(FakeDocumentProvider):
        def extract_structure(self, file_bytes, *, mime_type):
            raise RuntimeError("network timed out")

    tool = VisionExtractStructureTool(provider=_FailingProvider())
    result = tool.execute({"path": "doc.pdf"}, ctx)
    assert not result.success
    assert result.error.code == "structure_extraction_failed"
