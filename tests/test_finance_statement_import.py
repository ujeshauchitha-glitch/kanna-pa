from __future__ import annotations

from finance.imports.statement import import_statement
from finance.repository import CategoryRepository, TransactionRepository
from finance.service import ensure_default_categories
from vision.document.base import StatementExtraction, StatementTransaction
from vision.document.fake import FakeDocumentProvider


def _repos(db):
    ensure_default_categories(db)
    return TransactionRepository(db), CategoryRepository(db)


def test_happy_path_imports_only_debits(db):
    tx_repo, cat_repo = _repos(db)
    extraction = StatementExtraction(
        account_currency="INR",
        transactions=[
            StatementTransaction(date_text="2024-03-01", description="Cafe Coffee Day",
                                  amount_text="250.00", direction="debit"),
            StatementTransaction(date_text="2024-03-02", description="Salary", amount_text="50000",
                                  direction="credit"),
        ],
    )
    provider = FakeDocumentProvider(statement_extractions=[extraction])

    result = import_statement(b"x", mime_type="application/pdf", tx_repo=tx_repo, cat_repo=cat_repo,
                               provider=provider, default_currency="USD")

    assert result.created == 1
    assert len(result.errors) == 1
    assert "credit transaction" in result.errors[0].error
    tx = tx_repo.get(result.created_transaction_ids[0])
    assert tx.amount_minor == 25000
    assert tx.currency == "INR"  # account_currency used, not default_currency
    category = cat_repo.get(tx.category_id)
    assert category.name == "Food"  # "cafe" keyword


def test_unclear_direction_reported_not_imported(db):
    tx_repo, cat_repo = _repos(db)
    extraction = StatementExtraction(transactions=[
        StatementTransaction(description="Mystery row", amount_text="10", direction=None),
    ])
    provider = FakeDocumentProvider(statement_extractions=[extraction])

    result = import_statement(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                               provider=provider, default_currency="INR")

    assert result.created == 0
    assert "direction unclear" in result.errors[0].error
    assert tx_repo.search(limit=10) == []


def test_missing_amount_reported(db):
    tx_repo, cat_repo = _repos(db)
    extraction = StatementExtraction(transactions=[
        StatementTransaction(description="No amount row", direction="debit"),
    ])
    provider = FakeDocumentProvider(statement_extractions=[extraction])

    result = import_statement(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                               provider=provider, default_currency="INR")

    assert result.created == 0
    assert "missing amount" in result.errors[0].error


def test_unparseable_amount_reported(db):
    tx_repo, cat_repo = _repos(db)
    extraction = StatementExtraction(transactions=[
        StatementTransaction(description="Bad amount", amount_text="not a number", direction="debit"),
    ])
    provider = FakeDocumentProvider(statement_extractions=[extraction])

    result = import_statement(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                               provider=provider, default_currency="INR")

    assert result.created == 0
    assert len(result.errors) == 1


def test_missing_date_falls_back_to_today_with_note(db):
    tx_repo, cat_repo = _repos(db)
    extraction = StatementExtraction(transactions=[
        StatementTransaction(description="No date", amount_text="10", direction="debit"),
    ])
    provider = FakeDocumentProvider(statement_extractions=[extraction])

    result = import_statement(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                               provider=provider, default_currency="INR", today="2024-06-01")

    tx = tx_repo.get(result.created_transaction_ids[0])
    assert tx.occurred_at == "2024-06-01"
    assert "used today's date" in tx.notes


def test_currency_falls_back_to_default_when_no_account_currency(db):
    tx_repo, cat_repo = _repos(db)
    extraction = StatementExtraction(transactions=[
        StatementTransaction(description="Row", amount_text="20", direction="debit"),
    ])
    provider = FakeDocumentProvider(statement_extractions=[extraction])

    result = import_statement(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                               provider=provider, default_currency="USD")

    tx = tx_repo.get(result.created_transaction_ids[0])
    assert tx.currency == "USD"


def test_duplicate_row_is_skipped_not_double_counted(db):
    tx_repo, cat_repo = _repos(db)
    extraction = StatementExtraction(transactions=[
        StatementTransaction(date_text="2024-01-01", description="Store", amount_text="10",
                              direction="debit"),
    ])
    provider = FakeDocumentProvider(statement_extractions=[extraction, extraction])

    first = import_statement(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                              provider=provider, default_currency="INR")
    second = import_statement(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                               provider=provider, default_currency="INR")

    assert first.created == 1
    assert second.created == 0
    assert second.skipped_duplicates == 1
    assert len(tx_repo.search(limit=10)) == 1


def test_no_transactions_found_reports_document_level_error(db):
    tx_repo, cat_repo = _repos(db)
    provider = FakeDocumentProvider(statement_extractions=[
        StatementExtraction(notes="illegible scan")
    ])

    result = import_statement(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                               provider=provider, default_currency="INR")

    assert result.created == 0
    assert len(result.errors) == 1
    assert result.errors[0].row_number == 0
    assert "illegible scan" in result.errors[0].error


def test_provider_failure_is_reported_not_raised(db):
    tx_repo, cat_repo = _repos(db)

    def _boom(data, mime):
        raise RuntimeError("network error")

    provider = FakeDocumentProvider(statement_responder=_boom)
    result = import_statement(b"x", mime_type="image/png", tx_repo=tx_repo, cat_repo=cat_repo,
                               provider=provider, default_currency="INR")

    assert result.created == 0
    assert result.errors[0].row_number == 0
    assert "vision provider failed" in result.errors[0].error


def test_multiple_debits_all_imported(db):
    tx_repo, cat_repo = _repos(db)
    extraction = StatementExtraction(account_currency="INR", transactions=[
        StatementTransaction(date_text="2024-03-01", description="Uber", amount_text="100", direction="debit"),
        StatementTransaction(date_text="2024-03-02", description="Grocery store", amount_text="500", direction="debit"),
        StatementTransaction(date_text="2024-03-03", description="Movie ticket", amount_text="300", direction="debit"),
    ])
    provider = FakeDocumentProvider(statement_extractions=[extraction])

    result = import_statement(b"x", mime_type="application/pdf", tx_repo=tx_repo, cat_repo=cat_repo,
                               provider=provider, default_currency="INR")

    assert result.created == 3
    assert result.errors == []
