"""Deterministic natural-language extraction for the finance subsystem.

Nothing here is an LLM call. This module turns a sentence into a
structured `TransactionDraft` or `FinanceQuery` using regexes and date
arithmetic only — the same extraction runs identically every time, which
is what lets `finance/analytics.py` compute a total from the result
without ever asking a model to "remember" a number.

An `LLMPlanner` may *also* produce a `TransactionDraft`/`FinanceQuery` by
calling the model (useful for phrasing this module doesn't cover), but
either path lands in the same structured object before touching the
database — see `finance/service.py`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from finance.money import Money

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

_FILLER_PREFIXES = re.compile(
    r"^\s*(i\s+)?(spent|paid|bought|purchased|got)\b", re.IGNORECASE
)


def _today(timezone_name: str, now: datetime | None = None) -> date:
    if now is not None:
        return now.date()
    return datetime.now(ZoneInfo(timezone_name)).date()


def _iso(d: date) -> str:
    return d.isoformat()


# ---------------------------------------------------------------------------
# Transaction entry: "I spent ₹340 on lunch" / "paid 500 at Starbucks for coffee"
# ---------------------------------------------------------------------------

@dataclass
class TransactionDraft:
    amount_minor: int
    currency: str
    occurred_at: str  # YYYY-MM-DD
    description: str | None = None
    merchant: str | None = None
    category_hint: str | None = None  # raw keyword text; resolved against categories by the service layer


_MERCHANT_RE = re.compile(r"\bat\s+([A-Za-z0-9&'.]+(?:\s+[A-Za-z0-9&'.]+){0,2}?)(?=\s+(?:on|for|yesterday|today|$)|[.,!?]|$)", re.IGNORECASE)
_DESC_RE = re.compile(r"\b(?:on|for)\s+([A-Za-z0-9&'.]+(?:\s+[A-Za-z0-9&'.]+){0,3}?)(?=\s+(?:at|yesterday|today|last\s+\w+|on\s+\d)|[.,!?]|$)", re.IGNORECASE)
_DAYS_AGO_RE = re.compile(r"(\d+)\s+days?\s+ago", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def _extract_date(text: str, timezone_name: str, now: datetime | None = None) -> str:
    today = _today(timezone_name, now)
    lower = text.lower()

    iso_match = _ISO_DATE_RE.search(text)
    if iso_match:
        return iso_match.group(1)

    if "yesterday" in lower:
        return _iso(today - timedelta(days=1))
    if "today" in lower:
        return _iso(today)

    days_ago_match = _DAYS_AGO_RE.search(lower)
    if days_ago_match:
        return _iso(today - timedelta(days=int(days_ago_match.group(1))))

    for i, weekday in enumerate(_WEEKDAYS):
        if f"last {weekday}" in lower:
            days_since = (today.weekday() - i) % 7
            days_since = 7 if days_since == 0 else days_since
            return _iso(today - timedelta(days=days_since))
        if weekday in lower:
            days_since = (today.weekday() - i) % 7
            return _iso(today - timedelta(days=days_since))

    return _iso(today)


def parse_transaction(text: str, *, default_currency: str, timezone_name: str = "UTC",
                       now: datetime | None = None) -> TransactionDraft:
    """Extract amount, currency, date, merchant and a description from a spend sentence."""
    money = Money.parse(text, default_currency)
    occurred_at = _extract_date(text, timezone_name, now)

    merchant = None
    merchant_match = _MERCHANT_RE.search(text)
    if merchant_match:
        merchant = merchant_match.group(1).strip().rstrip(".,")

    description = None
    desc_match = _DESC_RE.search(text)
    if desc_match:
        description = desc_match.group(1).strip().rstrip(".,")
        # Don't let the description swallow a merchant clause introduced by "at".
        description = re.sub(r"\s+at\s+.*$", "", description, flags=re.IGNORECASE).strip()

    if description:
        description = description[0].upper() + description[1:]

    category_hint = description or merchant

    return TransactionDraft(
        amount_minor=money.amount_minor,
        currency=money.currency,
        occurred_at=occurred_at,
        description=description,
        merchant=merchant,
        category_hint=category_hint,
    )


# ---------------------------------------------------------------------------
# Queries: "How much did I spend on food this month?"
# ---------------------------------------------------------------------------

@dataclass
class FinanceQuery:
    kind: str  # "category_total" | "period_summary" | "search"
    start_date: str
    end_date: str
    category_hint: str | None = None
    currency: str | None = None
    text_contains: str | None = None


_QUERY_CATEGORY_RE = re.compile(
    r"\b(?:on|for)\s+([A-Za-z0-9&'.]+(?:\s+[A-Za-z0-9&'.]+){0,2}?)"
    r"(?=\s+(?:this|last|in|during|yesterday|today)\b|[?.,!]|$)",
    re.IGNORECASE,
)


def _period_bounds(text: str, timezone_name: str, now: datetime | None = None) -> tuple[str, str]:
    today = _today(timezone_name, now)
    lower = text.lower()

    if "this week" in lower:
        start = today - timedelta(days=today.weekday())
        return _iso(start), _iso(today)
    if "last month" in lower:
        first_of_this_month = today.replace(day=1)
        last_of_prev_month = first_of_this_month - timedelta(days=1)
        first_of_prev_month = last_of_prev_month.replace(day=1)
        return _iso(first_of_prev_month), _iso(last_of_prev_month)
    if "month" in lower and "last" not in lower:
        first_of_month = today.replace(day=1)
        return _iso(first_of_month), _iso(today)
    if "this year" in lower:
        return _iso(today.replace(month=1, day=1)), _iso(today)
    if "today" in lower:
        return _iso(today), _iso(today)
    if "yesterday" in lower:
        y = today - timedelta(days=1)
        return _iso(y), _iso(y)

    # Default: current month to date — the most common implicit meaning of
    # "how much did I spend" with no period stated.
    return _iso(today.replace(day=1)), _iso(today)


def parse_query(text: str, *, timezone_name: str = "UTC", now: datetime | None = None) -> FinanceQuery:
    start_date, end_date = _period_bounds(text, timezone_name, now)

    category_hint = None
    match = _QUERY_CATEGORY_RE.search(text)
    if match:
        candidate = match.group(1).strip().rstrip("?.,!")
        if candidate.lower() not in ("this", "that", "it", "everything", "all"):
            category_hint = candidate

    kind = "category_total" if category_hint else "period_summary"

    return FinanceQuery(kind=kind, start_date=start_date, end_date=end_date, category_hint=category_hint)
