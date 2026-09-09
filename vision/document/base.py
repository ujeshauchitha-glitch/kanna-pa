"""Structured document extraction: single receipts and multi-transaction statements.

Both `ReceiptExtraction` and `StatementExtraction` deliberately carry
*raw, unparsed* strings for date/currency/amount fields rather than
`date`/`Money` objects. Parsing "340.50" or "15/03/2024" into an exact
value is `finance`'s job (`finance.money.Money`, `finance.dates.
normalize_date`) — this module's job is only to read what's on the
page. Keeping that boundary means `vision` has no dependency on
`finance`, only the other way around (see `finance/imports/receipt.py`,
`finance/imports/statement.py`).
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


@dataclass
class StatementTransaction:
    """One row of a bank/card statement, as printed — not yet parsed."""

    date_text: str | None = None
    description: str | None = None
    amount_text: str | None = None
    # Statements often print debits and credits as separate columns, or a
    # single signed amount — the model is asked to normalize to one
    # amount plus this direction, since "amount_text" alone can't
    # disambiguate a statement that prints unsigned numbers in two columns.
    direction: str | None = None  # "debit" | "credit" | None if unclear


@dataclass
class StatementExtraction:
    account_currency: str | None = None
    transactions: list[StatementTransaction] = field(default_factory=list)
    raw_text: str = ""
    notes: str = ""


class DocumentProvider(Protocol):
    def extract_receipt(self, file_bytes: bytes, *, mime_type: str) -> ReceiptExtraction:
        ...

    def extract_statement(self, file_bytes: bytes, *, mime_type: str) -> StatementExtraction:
        ...
