"""Structured document extraction — currently scoped to receipts.

`ReceiptExtraction` deliberately carries *raw, unparsed* strings for the
date/currency/amount fields rather than `date`/`Money` objects. Parsing
"340.50" or "15/03/2024" into an exact value is `finance`'s job
(`finance.money.Money`, `finance.dates.normalize_date`) — this module's
job is only to read what's on the page. Keeping that boundary means
`vision` has no dependency on `finance`, only the other way around (see
`finance/imports/receipt.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LineItem:
    description: str
    amount_text: str | None = None


@dataclass
class ReceiptExtraction:
    merchant: str | None = None
    date_text: str | None = None
    currency_code: str | None = None
    total_amount_text: str | None = None
    line_items: list[LineItem] = field(default_factory=list)
    raw_text: str = ""
    # Honest caveats surfaced from extraction — e.g. "no total found",
    # "date printed unclearly, best guess" — carried through to the
    # transaction's notes rather than silently dropped.
    notes: str = ""


class DocumentProvider(Protocol):
    def extract_receipt(self, file_bytes: bytes, *, mime_type: str) -> ReceiptExtraction:
        ...
