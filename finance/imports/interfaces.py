"""Interfaces for import sources.

`csv_import.py` and, as of Phase 2, `receipt.py` are real implementations.
Bank statement files (beyond a plain CSV export, already covered by
`csv_import.import_csv`) still need document-layout understanding beyond
single-receipt extraction, so only the shape is defined here for that one
— there is deliberately no fake/stub implementation that would silently
produce made-up transactions.
"""
from __future__ import annotations

from typing import Protocol

from finance.imports.csv_import import ImportResult

# `finance.imports.receipt.import_receipt` is the real implementation —
# a module-level function taking explicit repositories and a
# `vision.document.base.DocumentProvider`, following the same style as
# `csv_import.import_csv` rather than this Protocol's method-on-an-object
# shape. It's kept for reference/future alternate implementations, but
# nothing in `finance` currently constructs a `ReceiptImporter` directly.
class ReceiptImporter(Protocol):
    """Extracts a transaction from a photographed/scanned receipt.

    See `finance.imports.receipt.import_receipt` for the real, in-use
    implementation (Anthropic-vision-backed by default via
    `vision.document.anthropic_document.AnthropicDocumentProvider`).
    """

    def import_receipt(self, image_bytes: bytes, *, default_currency: str) -> ImportResult: ...


class StatementImporter(Protocol):
    """Extracts transactions from a bank/card statement file (PDF or CSV export).

    Never requires bank login credentials — only a file the user already
    has (a downloaded statement). A PDF statement importer needs
    multi-page/multi-transaction document understanding beyond what
    `vision.document`'s single-receipt extraction does today; the CSV
    case is already covered by `finance.imports.csv_import.import_csv`.
    """

    def import_statement(self, file_bytes: bytes, *, default_currency: str) -> ImportResult: ...
