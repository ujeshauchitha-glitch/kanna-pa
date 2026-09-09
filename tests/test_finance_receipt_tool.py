from __future__ import annotations

from finance.tools import FinanceImportReceiptTool
from vision.document.base import ReceiptExtraction
from vision.document.fake import FakeDocumentProvider


def test_tool_reads_sandboxed_file_and_logs_transaction(ctx, tmp_path):
    (tmp_path / "receipt.png").write_bytes(b"\x89PNG fake bytes")
    provider = FakeDocumentProvider(extractions=[ReceiptExtraction(
        merchant="Cafe Coffee Day", date_text="2024-03-15", total_amount_text="150",
    )])
    tool = FinanceImportReceiptTool(provider=provider)

    result = tool.execute({"path": "receipt.png"}, ctx)

    assert result.success
    assert result.data["created"] == 1
    assert result.data["transaction_id"]
    assert provider.calls[0][1] == "image/png"


def test_tool_rejects_missing_file(ctx):
    tool = FinanceImportReceiptTool(provider=FakeDocumentProvider())
    result = tool.execute({"path": "does-not-exist.png"}, ctx)
    assert not result.success
    assert result.error.code == "not_found"


def test_tool_rejects_sandbox_escape(ctx):
    tool = FinanceImportReceiptTool(provider=FakeDocumentProvider())
    result = tool.execute({"path": "../../etc/passwd"}, ctx)
    assert not result.success
    assert result.error.code == "sandbox_violation"


def test_tool_rejects_unsupported_extension(ctx, tmp_path):
    (tmp_path / "receipt.txt").write_text("not an image")
    tool = FinanceImportReceiptTool(provider=FakeDocumentProvider())
    result = tool.execute({"path": "receipt.txt"}, ctx)
    assert not result.success
    assert result.error.code == "unsupported_file_type"


def test_tool_reports_extraction_failure(ctx, tmp_path):
    (tmp_path / "receipt.png").write_bytes(b"fake")
    provider = FakeDocumentProvider(extractions=[ReceiptExtraction(notes="blurry")])
    tool = FinanceImportReceiptTool(provider=provider)
    result = tool.execute({"path": "receipt.png"}, ctx)
    assert not result.success
    assert result.error.code == "receipt_extraction_failed"


def test_tool_honors_explicit_mime_type_override(ctx, tmp_path):
    (tmp_path / "receipt.bin").write_bytes(b"fake pdf bytes")
    provider = FakeDocumentProvider(extractions=[ReceiptExtraction(total_amount_text="10")])
    tool = FinanceImportReceiptTool(provider=provider)
    result = tool.execute({"path": "receipt.bin", "mime_type": "application/pdf"}, ctx)
    assert result.success
    assert provider.calls[0][1] == "application/pdf"


def test_tool_reports_duplicate(ctx, tmp_path):
    (tmp_path / "receipt.png").write_bytes(b"fake")
    extraction = ReceiptExtraction(date_text="2024-01-01", total_amount_text="5")
    provider = FakeDocumentProvider(extractions=[extraction, extraction])
    tool = FinanceImportReceiptTool(provider=provider)

    first = tool.execute({"path": "receipt.png"}, ctx)
    second = tool.execute({"path": "receipt.png"}, ctx)

    assert first.success and first.data["created"] == 1
    assert second.success and second.data["skipped_duplicate"] == "True"
