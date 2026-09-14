from __future__ import annotations

from vision.document.anthropic_document import parse_statement_json
from vision.document.base import StatementExtraction, StatementTransaction
from vision.document.fake import FakeDocumentProvider


def test_fake_document_provider_statement_extraction():
    extraction = StatementExtraction(
        account_currency="INR",
        transactions=[StatementTransaction(description="Coffee", amount_text="250", direction="debit")],
    )
    provider = FakeDocumentProvider(statement_extractions=[extraction])
    result = provider.extract_statement(b"bytes", mime_type="application/pdf")
    assert result.account_currency == "INR"
    assert result.transactions[0].description == "Coffee"
    assert provider.statement_calls == [(b"bytes", "application/pdf")]


def test_fake_document_provider_statement_default_is_empty():
    provider = FakeDocumentProvider()
    result = provider.extract_statement(b"x", mime_type="application/pdf")
    assert result == StatementExtraction()


# -- parse_statement_json: pure JSON -> StatementExtraction conversion --

def test_parse_statement_json_full():
    data = {
        "account_currency": "INR",
        "transactions": [
            {"date": "2024-03-01", "description": "Coffee Shop", "amount": "250.00", "direction": "debit"},
            {"date": "2024-03-02", "description": "Salary", "amount": "50000.00", "direction": "credit"},
        ],
        "notes": "",
    }
    result = parse_statement_json(data, raw_text="raw ocr text")
    assert result.account_currency == "INR"
    assert len(result.transactions) == 2
    assert result.transactions[0] == StatementTransaction(
        date_text="2024-03-01", description="Coffee Shop", amount_text="250.00", direction="debit")
    assert result.transactions[1].direction == "credit"
    assert result.raw_text == "raw ocr text"


def test_parse_statement_json_all_null():
    data = {"account_currency": None, "transactions": [], "notes": "statement too blurry to read"}
    result = parse_statement_json(data)
    assert result.account_currency is None
    assert result.transactions == []
    assert result.notes == "statement too blurry to read"


def test_parse_statement_json_tolerates_wrong_types():
    data = {"account_currency": 123, "transactions": "not a list"}
    result = parse_statement_json(data)
    assert result.account_currency is None
    assert result.transactions == []
    assert "account_currency" in result.notes
    assert "transactions" in result.notes


def test_parse_statement_json_skips_malformed_rows_keeps_valid_ones():
    data = {"transactions": [
        {"description": "ok row", "amount": "5", "direction": "debit"},
        "junk",
        {"no_useful_fields": True},
    ]}
    result = parse_statement_json(data)
    # "junk" is skipped with a note; the malformed-but-dict row still
    # produces an (empty) StatementTransaction rather than being dropped,
    # since a row missing fields isn't the same failure as a row that
    # isn't even an object — downstream (finance) rejects it for lacking
    # an amount.
    assert len(result.transactions) == 2
    assert result.transactions[0].description == "ok row"
    assert "transaction 1" in result.notes


def test_parse_statement_json_rejects_unrecognized_direction():
    data = {"transactions": [{"description": "row", "amount": "1", "direction": "sideways"}]}
    result = parse_statement_json(data)
    assert result.transactions[0].direction is None
    assert "unrecognized direction" in result.notes
