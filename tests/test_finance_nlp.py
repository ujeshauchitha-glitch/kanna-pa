from __future__ import annotations

from datetime import datetime, timezone

from finance import nlp


def test_spec_example_i_spent_340_on_lunch():
    now = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
    draft = nlp.parse_transaction("I spent ₹340 on lunch", default_currency="USD",
                                   timezone_name="UTC", now=now)
    assert draft.amount_minor == 34000
    assert draft.currency == "INR"
    assert draft.description == "Lunch"
    assert draft.occurred_at == "2024-03-15"


def test_merchant_extraction():
    now = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
    draft = nlp.parse_transaction("I paid 500 at Starbucks for coffee", default_currency="INR",
                                   timezone_name="UTC", now=now)
    assert draft.amount_minor == 50000
    assert draft.merchant == "Starbucks"
    assert draft.description == "Coffee"


def test_yesterday_date():
    now = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
    draft = nlp.parse_transaction("I spent 100 on snacks yesterday", default_currency="INR",
                                   timezone_name="UTC", now=now)
    assert draft.occurred_at == "2024-03-14"


def test_explicit_iso_date():
    now = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
    draft = nlp.parse_transaction("Paid 2000 on 2024-01-05 for rent", default_currency="INR",
                                   timezone_name="UTC", now=now)
    assert draft.occurred_at == "2024-01-05"


def test_default_currency_used_without_symbol():
    draft = nlp.parse_transaction("spent 50 on tea", default_currency="INR")
    assert draft.currency == "INR"


def test_query_this_month_default_period():
    now = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
    query = nlp.parse_query("How much did I spend on food this month?", timezone_name="UTC", now=now)
    assert query.start_date == "2024-03-01"
    assert query.end_date == "2024-03-15"
    assert query.category_hint == "food"
    assert query.kind == "category_total"


def test_query_last_month():
    now = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
    query = nlp.parse_query("How much did I spend last month?", timezone_name="UTC", now=now)
    assert query.start_date == "2024-02-01"
    assert query.end_date == "2024-02-29"  # 2024 is a leap year


def test_query_today():
    now = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
    query = nlp.parse_query("How much did I spend today?", timezone_name="UTC", now=now)
    assert query.start_date == "2024-03-15"
    assert query.end_date == "2024-03-15"


def test_query_without_period_defaults_to_month_to_date():
    now = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
    query = nlp.parse_query("How much did I spend overall?", timezone_name="UTC", now=now)
    assert query.start_date == "2024-03-01"
    assert query.end_date == "2024-03-15"
    assert query.category_hint is None
    assert query.kind == "period_summary"
