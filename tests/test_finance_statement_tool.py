from __future__ import annotations

from finance.tools import FinanceImportStatementTool
from vision.document.base import StatementExtraction, StatementTransaction
from vision.document.fake import FakeDocumentProvider


def test_tool_reads_sandboxed_file_and_imports_debits(ctx, tmp_path):
    (tmp_path / "statement.pdf").write_bytes(b"%PDF fake bytes")
    extraction = StatementExtraction(account_currency="INR", transactions=[
        StatementTransaction(date_text="2024-03-01", description="Cafe Coffee Day",
                              amount_text="150", direction="debit"),
        StatementTransaction(date_text="2024-03-02", description="Salary", amount_text="50000",
                              direction="credit"),
    ])
    provider = FakeDocumentProvider(statement_extractions=[extraction])
    tool = FinanceImportStatementTool(provider=provider)

    result = tool.execute({"path": "statement.pdf"}, ctx)

    assert result.success
    assert result.data["created"] == 1
    assert result.data["error_count"] == 1
    assert provider.statement_calls[0][1] == "application/pdf"


def test_tool_rejects_missing_file(ctx):
    tool = FinanceImportStatementTool(provider=FakeDocumentProvider())
    result = tool.execute({"path": "does-not-exist.pdf"}, ctx)
    assert not result.success
    assert result.error.code == "not_found"


def test_tool_rejects_sandbox_escape(ctx):
    tool = FinanceImportStatementTool(provider=FakeDocumentProvider())
    result = tool.execute({"path": "../../etc/passwd"}, ctx)
    assert not result.success
    assert result.error.code == "sandbox_violation"


def test_tool_rejects_unsupported_extension(ctx, tmp_path):
    (tmp_path / "statement.txt").write_text("not a document")
    tool = FinanceImportStatementTool(provider=FakeDocumentProvider())
    result = tool.execute({"path": "statement.txt"}, ctx)
    assert not result.success
    assert result.error.code == "unsupported_file_type"


def test_tool_fails_cleanly_when_no_transactions_found(ctx, tmp_path):
    (tmp_path / "statement.pdf").write_bytes(b"fake")
    provider = FakeDocumentProvider(statement_extractions=[StatementExtraction(notes="blurry scan")])
    tool = FinanceImportStatementTool(provider=provider)

    result = tool.execute({"path": "statement.pdf"}, ctx)

    assert not result.success
    assert result.error.code == "statement_extraction_failed"


def test_tool_succeeds_even_when_every_row_is_a_credit(ctx, tmp_path):
    """A statement full of credits (no debits at all) is a legitimate,
    successful read — not a tool failure — it just imports nothing."""
    (tmp_path / "statement.pdf").write_bytes(b"fake")
    extraction = StatementExtraction(transactions=[
        StatementTransaction(description="Refund", amount_text="10", direction="credit"),
    ])
    provider = FakeDocumentProvider(statement_extractions=[extraction])
    tool = FinanceImportStatementTool(provider=provider)

    result = tool.execute({"path": "statement.pdf"}, ctx)

    assert result.success
    assert result.data["created"] == 0
    assert result.data["error_count"] == 1


def test_tool_honors_explicit_mime_type_override(ctx, tmp_path):
    (tmp_path / "statement.bin").write_bytes(b"fake")
    extraction = StatementExtraction(transactions=[
        StatementTransaction(description="Row", amount_text="10", direction="debit"),
    ])
    provider = FakeDocumentProvider(statement_extractions=[extraction])
    tool = FinanceImportStatementTool(provider=provider)

    result = tool.execute({"path": "statement.bin", "mime_type": "image/jpeg"}, ctx)

    assert result.success
    assert provider.statement_calls[0][1] == "image/jpeg"
