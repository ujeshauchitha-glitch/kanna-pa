from __future__ import annotations

from finance.export import to_csv, to_json
from finance.imports.csv_import import import_csv
from finance.service import FinanceService


CSV_TEXT = """date,amount,description,merchant,category
2024-03-01,340,Lunch,Cafe Coffee Day,Food
2024-03-02,1200,Groceries,BigBasket,Food
not-a-date,500,Broken row,,
"""


def test_import_creates_transactions_and_reports_errors(db):
    svc = FinanceService(db, default_currency="INR")
    result = import_csv(CSV_TEXT, tx_repo=svc.transactions, cat_repo=svc.categories,
                         default_currency="INR")

    assert result.created == 2
    assert len(result.errors) == 1
    assert result.errors[0].row_number == 4


def test_import_dedupes_identical_rows(db):
    svc = FinanceService(db, default_currency="INR")
    first = import_csv(CSV_TEXT, tx_repo=svc.transactions, cat_repo=svc.categories,
                        default_currency="INR")
    second = import_csv(CSV_TEXT, tx_repo=svc.transactions, cat_repo=svc.categories,
                         default_currency="INR")

    assert first.created == 2
    assert second.created == 0
    assert second.skipped_duplicates == 2


def test_export_round_trip(db):
    svc = FinanceService(db, default_currency="INR")
    import_csv(CSV_TEXT, tx_repo=svc.transactions, cat_repo=svc.categories, default_currency="INR")

    transactions = svc.search(limit=100)
    csv_out = to_csv(transactions)
    assert "340" in csv_out
    assert "Lunch" in csv_out

    json_out = to_json(transactions)
    import json
    parsed = json.loads(json_out)
    assert len(parsed) == 2
    assert {p["amount_minor"] for p in parsed} == {34000, 120000}


def test_custom_column_mapping(db):
    svc = FinanceService(db, default_currency="INR")
    custom_csv = "when,how_much,what\n2024-03-01,99,Coffee\n"
    result = import_csv(custom_csv, tx_repo=svc.transactions, cat_repo=svc.categories,
                         default_currency="INR",
                         column_map={"date": "when", "amount": "how_much", "description": "what"})
    assert result.created == 1
    tx = svc.search(limit=1)[0]
    assert tx.amount_minor == 9900
    assert tx.description == "Coffee"
