"""Structured document extraction: receipts, statements, and generic structure.

`ReceiptExtraction` and `StatementExtraction` deliberately carry *raw,
unparsed* strings for date/currency/amount fields rather than
`date`/`Money` objects. Parsing "340.50" or "15/03/2024" into an exact
value is `finance`'s job (`finance.money.Money`, `finance.dates.
normalize_date`) — this module's job is only to read what's on the
page. Keeping that boundary means `vision` has no dependency on
`finance`, only the other way around (see `finance/imports/receipt.py`,
`finance/imports/statement.py`).

`DocumentStructure` generalizes past the fixed receipt/statement-row
shapes to arbitrary sections/headings/paragraphs/tables — for reading a
document (an assignment PDF, an article, a report) that isn't either of
those two specific things.
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


@dataclass
class StructureTable:
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class StructureSection:
    """One heading-and-its-content unit of an arbitrary document.

    Deliberately shaped like `documents.model.Section` (heading/level/
    paragraphs/bullets/table) so a caller can map a `DocumentStructure`
    into a `documents.model.Document` and re-render it — but this module
    doesn't import `documents` itself, the same one-way-dependency
    discipline `ReceiptExtraction`/`StatementExtraction` already follow
    with `finance`. The mapping, if a caller wants it, lives on their
    side.
    """

    heading: str | None = None
    level: int = 1
    paragraphs: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    table: StructureTable | None = None


@dataclass
class DocumentStructure:
    """Arbitrary document structure — sections/headings/paragraphs/tables,
    not a fixed receipt or statement-row shape. What a PDF assignment,
    article, or report reader needs instead of `ReceiptExtraction`'s or
    `StatementExtraction`'s narrow schema.
    """

    title: str | None = None
    sections: list[StructureSection] = field(default_factory=list)
    raw_text: str = ""
    notes: str = ""


class DocumentProvider(Protocol):
    def extract_receipt(self, file_bytes: bytes, *, mime_type: str) -> ReceiptExtraction:
        ...

    def extract_statement(self, file_bytes: bytes, *, mime_type: str) -> StatementExtraction:
        ...

    def extract_structure(self, file_bytes: bytes, *, mime_type: str) -> DocumentStructure:
        ...
