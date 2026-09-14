"""Interfaces for import sources.

`csv_import.py`, `receipt.py`, and, as of this pass, `statement.py` are
all real implementations. Both `ReceiptImporter` and `StatementImporter`
below are kept only as documented Protocol shapes — the actual code is
module-level functions (`import_receipt`, `import_statement`) taking
explicit repositories and a `vision.document.base.DocumentProvider`,
following `csv_import.import_csv`'s style rather than a method-on-an-
object shape. Nothing in `finance` constructs either Protocol directly.
"""
from __future__ import annotations

from typing import Protocol

from finance.imports.csv_import import ImportResult


class ReceiptImporter(Protocol):
    """Extracts a transaction from a photographed/scanned receipt.

    See `finance.imports.receipt.import_receipt` for the real, in-use
    implementation (Anthropic-vision-backed by default via
    `vision.document.anthropic_document.AnthropicDocumentProvider`).
    """

    def import_receipt(self, image_bytes: bytes, *, default_currency: str) -> ImportResult: ...


class StatementImporter(Protocol):
    """Extracts transactions from a bank/card statement file (image or PDF, possibly multi-page).

    Never requires bank login credentials — only a file the user already
    has (a downloaded statement). See `finance.imports.statement.
    import_statement` for the real, in-use implementation — it imports
    debit (spend) rows only; see that module's docstring for why credit
    rows are reported, not silently dropped or misrepresented as spend.
    A plain CSV export of a statement is already covered by
    `finance.imports.csv_import.import_csv` instead.
    """

    def import_statement(self, file_bytes: bytes, *, default_currency: str) -> ImportResult: ...
