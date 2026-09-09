"""Date-string normalization shared by every finance import path.

Factored out of `imports/csv_import.py` so `imports/receipt.py` (receipt
dates come back as whatever format was printed — "15/03/2024", "Mar 15,
2024", ...) doesn't duplicate it.
"""
from __future__ import annotations

from datetime import datetime

_FORMATS = (
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
    "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y",
)


def normalize_date(raw: str) -> str:
    """Parse a date string in any of several common formats to `YYYY-MM-DD`.

    Raises `ValueError` for anything unrecognized — callers decide how to
    handle that (CSV import reports it as a per-row error; receipt import
    falls back to today's date with a note, since a missing/unreadable
    date shouldn't block logging the spend).
    """
    cleaned = raw.strip()
    for fmt in _FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unrecognized date format: {raw!r}")
