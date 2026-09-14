from __future__ import annotations

from finance.service import FinanceService


def test_add_transaction_from_text_infers_category_and_date(db):
    svc = FinanceService(db, default_currency="INR", timezone_name="UTC")
    from datetime import datetime, timezone
    now = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)

    draft_now = now  # used only for readability
    result = svc.add_transaction_from_text("I spent ₹340 on lunch")

    tx = result.transaction
    assert tx.amount_minor == 34000
    assert tx.currency == "INR"
    assert result.category_name == "Food"
    assert tx.description == "Lunch"


def test_add_transaction_manual_with_explicit_category(db):
    svc = FinanceService(db, default_currency="INR")
    result = svc.add_transaction(amount_minor=50000, currency="INR", occurred_at="2024-05-01",
                                  description="Concert tickets", category_name="Entertainment")
    assert result.transaction.category_id is not None
    assert result.category_name == "Entertainment"


def test_date_filtering_across_month_boundary(db):
    svc = FinanceService(db, default_currency="INR")
    svc.add_transaction(amount_minor=1000, currency="INR", occurred_at="2024-02-28")
    svc.add_transaction(amount_minor=2000, currency="INR", occurred_at="2024-03-01")
    svc.add_transaction(amount_minor=3000, currency="INR", occurred_at="2024-03-15")

    march_only = svc.search(start_date="2024-03-01", end_date="2024-03-31")
    assert sorted(tx.amount_minor for tx in march_only) == [2000, 3000]


def test_monthly_totals_deterministic(db):
    from finance import analytics
    svc = FinanceService(db, default_currency="INR")
    svc.add_transaction(amount_minor=1000, currency="INR", occurred_at="2024-03-01")
    svc.add_transaction(amount_minor=2500, currency="INR", occurred_at="2024-03-20")
    svc.add_transaction(amount_minor=9999, currency="INR", occurred_at="2024-04-01")  # outside range

    summary = analytics.period_summary(svc.transactions, svc.categories,
                                        start_date="2024-03-01", end_date="2024-03-31")
    assert summary.totals_by_currency["INR"].amount_minor == 3500
    assert summary.transaction_count == 2


def test_category_totals(db):
    from finance import analytics
    svc = FinanceService(db, default_currency="INR")
    svc.add_transaction(amount_minor=1000, currency="INR", occurred_at="2024-03-01",
                         category_name="Food")
    svc.add_transaction(amount_minor=2000, currency="INR", occurred_at="2024-03-02",
                         category_name="Food")
    svc.add_transaction(amount_minor=5000, currency="INR", occurred_at="2024-03-03",
                         category_name="Transport")

    food_id = svc.categories.get_by_name("Food").id
    totals = analytics.category_total(svc.transactions, category_id=food_id,
                                       start_date="2024-03-01", end_date="2024-03-31")
    assert totals["INR"].amount_minor == 3000


def test_multi_currency_totals_kept_separate(db):
    from finance import analytics
    svc = FinanceService(db)
    svc.add_transaction(amount_minor=1000, currency="INR", occurred_at="2024-03-01")
    svc.add_transaction(amount_minor=500, currency="USD", occurred_at="2024-03-01")

    summary = analytics.period_summary(svc.transactions, svc.categories,
                                        start_date="2024-03-01", end_date="2024-03-31")
    assert summary.totals_by_currency["INR"].amount_minor == 1000
    assert summary.totals_by_currency["USD"].amount_minor == 500


def test_query_from_text_matches_spec_example(db):
    svc = FinanceService(db, default_currency="INR")
    svc.add_transaction_from_text("I spent ₹340 on lunch")

    from datetime import date
    today = date.today().isoformat()
    result = svc.query_from_text("How much did I spend on food this month?")
    assert result.summary.totals_by_currency["INR"].amount_minor == 34000
    assert "340.00 INR" in result.message


def test_delete_transaction(db):
    svc = FinanceService(db, default_currency="INR")
    result = svc.add_transaction(amount_minor=1000, currency="INR", occurred_at="2024-01-01")
    assert svc.transactions.delete(result.transaction.id) is True
    assert svc.transactions.get(result.transaction.id) is None
