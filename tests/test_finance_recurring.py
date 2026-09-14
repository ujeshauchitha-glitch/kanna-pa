from __future__ import annotations

from datetime import date

import pytest

from finance.recurring import next_occurrence
from finance.service import FinanceService


def test_daily():
    assert next_occurrence(date(2024, 3, 1), "daily") == date(2024, 3, 2)


def test_weekly():
    assert next_occurrence(date(2024, 3, 1), "weekly") == date(2024, 3, 8)


def test_biweekly():
    assert next_occurrence(date(2024, 3, 1), "biweekly") == date(2024, 3, 15)


def test_monthly_regular():
    assert next_occurrence(date(2024, 3, 15), "monthly") == date(2024, 4, 15)


def test_monthly_end_of_month_rollover_to_shorter_month():
    # Jan 31 -> Feb 28 (2025 is not a leap year)
    assert next_occurrence(date(2025, 1, 31), "monthly") == date(2025, 2, 28)


def test_monthly_end_of_month_rollover_leap_year():
    assert next_occurrence(date(2024, 1, 31), "monthly") == date(2024, 2, 29)


def test_monthly_december_rolls_into_next_year():
    assert next_occurrence(date(2024, 12, 15), "monthly") == date(2025, 1, 15)


def test_yearly():
    assert next_occurrence(date(2024, 3, 1), "yearly") == date(2025, 3, 1)


def test_unknown_frequency_raises():
    with pytest.raises(ValueError):
        next_occurrence(date(2024, 1, 1), "fortnightly")


def test_apply_due_recurring_creates_transaction_and_advances(db):
    svc = FinanceService(db, default_currency="INR")
    rec = svc.add_recurring(description="Netflix", amount_minor=49900, currency="INR",
                             frequency="monthly", next_occurrence="2024-03-01",
                             category_name="Entertainment")

    created = svc.apply_due_recurring(as_of_date="2024-03-01")
    assert len(created) == 1
    assert created[0].amount_minor == 49900
    assert created[0].source == "recurring"

    updated = svc.recurring.list()[0]
    assert updated.next_occurrence == "2024-04-01"

    # Not due again immediately.
    assert svc.apply_due_recurring(as_of_date="2024-03-15") == []
