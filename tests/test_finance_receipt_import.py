from __future__ import annotations

from finance.imports.receipt import import_receipt
from finance.repository import CategoryRepository, TransactionRepository
from finance.service import ensure_default_categories
from vision.document.base import LineItem, ReceiptExtraction
from vision.document.fake import FakeDocumentProvider


def _repos(db):
    ensure_default_categories(db)
    return TransactionRepository(db), CategoryRepository(db)


def test_happy_path_creates_transaction_with_category(db):
    tx_repo, cat_repo = _repos(db)
    provider = FakeDocumentProvider(extractions=[ReceiptExtraction(
        merchant="Cafe Coffee Day", date_text="2024-03-15", currency_code="INR",
        total_amount_text="340.00",
        line_items=[LineItem("Latte", "340.00")], raw_text="...",
    )])

    result = import_receipt(b"fake-image-bytes", mime_type="image/jpeg", tx_repo=tx_repo,
                             cat_repo=cat_repo, provider=provider, default_currency="INR")

    assert result.created == 1
    assert result.errors == []
    tx = tx_repo.get(result.created_transaction_ids[0])
    assert tx.amount_minor == 34000
    assert tx.currency == "INR"
    assert tx.occurred_at == "2024-03-15"
    assert tx.merchant == "Cafe Coffee Day"
    assert tx.source == "receipt_ocr"
    category = cat_repo.get(tx.category_id)
    assert category.name == "Food"  # "cafe" keyword matches Food


def test_missing_amount_reports_error_and_creates_nothing(db):
    tx_repo, cat_repo = _repos(db)
    provider = FakeDocumentProvider(extractions=[ReceiptExtraction(
        merchant="Unknown Store", notes="total illegible",
    )])

    result = import_receipt(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                             provider=provider, default_currency="INR")

    assert result.created == 0
    assert len(result.errors) == 1
    assert "total illegible" in result.errors[0].error
    assert tx_repo.search(limit=10) == []


def test_missing_date_defaults_to_today_with_note(db):
    tx_repo, cat_repo = _repos(db)
    provider = FakeDocumentProvider(extractions=[ReceiptExtraction(total_amount_text="99")])

    result = import_receipt(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                             provider=provider, default_currency="INR", today="2024-06-01")

    assert result.created == 1
    tx = tx_repo.get(result.created_transaction_ids[0])
    assert tx.occurred_at == "2024-06-01"
    assert "used today's date" in tx.notes


def test_unparseable_date_falls_back_to_today_with_note(db):
    tx_repo, cat_repo = _repos(db)
    provider = FakeDocumentProvider(extractions=[ReceiptExtraction(
        total_amount_text="50", date_text="not a real date",
    )])

    result = import_receipt(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                             provider=provider, default_currency="INR", today="2024-06-01")

    tx = tx_repo.get(result.created_transaction_ids[0])
    assert tx.occurred_at == "2024-06-01"
    assert "could not parse receipt date" in tx.notes


def test_currency_falls_back_to_default_when_not_extracted(db):
    tx_repo, cat_repo = _repos(db)
    provider = FakeDocumentProvider(extractions=[ReceiptExtraction(total_amount_text="20")])

    result = import_receipt(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                             provider=provider, default_currency="USD")

    tx = tx_repo.get(result.created_transaction_ids[0])
    assert tx.currency == "USD"
    assert tx.amount_minor == 2000


def test_duplicate_receipt_is_skipped_not_double_counted(db):
    tx_repo, cat_repo = _repos(db)
    extraction = ReceiptExtraction(merchant="Store", date_text="2024-01-01", total_amount_text="10")
    provider = FakeDocumentProvider(extractions=[extraction, extraction])

    first = import_receipt(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                            provider=provider, default_currency="INR")
    second = import_receipt(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                             provider=provider, default_currency="INR")

    assert first.created == 1
    assert second.created == 0
    assert second.skipped_duplicates == 1
    assert len(tx_repo.search(limit=10)) == 1


def test_provider_failure_is_reported_not_raised(db):
    tx_repo, cat_repo = _repos(db)

    def _boom(data, mime):
        raise RuntimeError("network error")

    provider = FakeDocumentProvider(responder=_boom)
    result = import_receipt(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                             provider=provider, default_currency="INR")

    assert result.created == 0
    assert "vision provider failed" in result.errors[0].error


def test_unparseable_amount_reports_error(db):
    tx_repo, cat_repo = _repos(db)
    provider = FakeDocumentProvider(extractions=[ReceiptExtraction(total_amount_text="not a number")])

    result = import_receipt(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                             provider=provider, default_currency="INR")

    assert result.created == 0
    assert len(result.errors) == 1
