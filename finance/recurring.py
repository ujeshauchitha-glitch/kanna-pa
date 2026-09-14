"""Recurring-expense occurrence math.

Kept separate from `repository.py` because the date arithmetic (in
particular month-end rollover, e.g. Jan 31 -> Feb 28/29) is worth testing
in isolation from the database.
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta

_FREQUENCIES = {"daily", "weekly", "biweekly", "monthly", "yearly"}


def _add_months(d: date, months: int) -> date:
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(d.day, last_day)
    return date(year, month, day)


def next_occurrence(current: date, frequency: str) -> date:
    """Return the next date after `current` for the given frequency."""
    if frequency not in _FREQUENCIES:
        raise ValueError(f"unknown recurrence frequency: {frequency!r}")

    if frequency == "daily":
        return current + timedelta(days=1)
    if frequency == "weekly":
        return current + timedelta(weeks=1)
    if frequency == "biweekly":
        return current + timedelta(weeks=2)
    if frequency == "monthly":
        return _add_months(current, 1)
    if frequency == "yearly":
        return _add_months(current, 12)
    raise AssertionError("unreachable")  # pragma: no cover
