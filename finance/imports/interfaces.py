"""Interfaces for import sources not yet implemented in Phase 1.

`csv_import.py` is the one real implementation right now. Receipt images
and bank statement files need OCR/document-parsing (see `vision/` in the
roadmap) before a real implementation can exist, so only the shape is
defined here — there is deliberately no fake/stub implementation that
would silently produce made-up transactions.
"""
from __future__ import annotations

from typing import Protocol

from finance.imports.csv_import import ImportResult


class ReceiptImporter(Protocol):
    """Extracts a transaction draft from a photographed/scanned receipt.

    Requires OCR + document understanding (Phase 2+, see `vision/`).
    """

    def import_receipt(self, image_bytes: bytes, *, default_currency: str) -> ImportResult: ...


class StatementImporter(Protocol):
    """Extracts transactions from a bank/card statement file (PDF or CSV export).

    Never requires bank login credentials — only a file the user already
    has (a downloaded statement). A PDF statement importer needs
    `vision/document` (Phase 2+); the CSV case is already covered by
    `finance.imports.csv_import.import_csv`.
    """

    def import_statement(self, file_bytes: bytes, *, default_currency: str) -> ImportResult: ...
