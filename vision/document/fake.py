"""A scripted document provider for tests — no network."""
from __future__ import annotations

from collections.abc import Callable

from vision.document.base import ReceiptExtraction, StatementExtraction


class FakeDocumentProvider:
    def __init__(self, extractions: list[ReceiptExtraction] | None = None,
                 responder: Callable[[bytes, str], ReceiptExtraction] | None = None,
                 statement_extractions: list[StatementExtraction] | None = None,
                 statement_responder: Callable[[bytes, str], StatementExtraction] | None = None) -> None:
        self._extractions = list(extractions or [])
        self._responder = responder
        self._statement_extractions = list(statement_extractions or [])
        self._statement_responder = statement_responder
        self.calls: list[tuple[bytes, str]] = []
        self.statement_calls: list[tuple[bytes, str]] = []

    def extract_receipt(self, file_bytes: bytes, *, mime_type: str) -> ReceiptExtraction:
        self.calls.append((file_bytes, mime_type))
        if self._responder is not None:
            return self._responder(file_bytes, mime_type)
        if self._extractions:
            return self._extractions.pop(0)
        return ReceiptExtraction()

    def extract_statement(self, file_bytes: bytes, *, mime_type: str) -> StatementExtraction:
        self.statement_calls.append((file_bytes, mime_type))
        if self._statement_responder is not None:
            return self._statement_responder(file_bytes, mime_type)
        if self._statement_extractions:
            return self._statement_extractions.pop(0)
        return StatementExtraction()
