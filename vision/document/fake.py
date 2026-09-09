"""A scripted document provider for tests — no network."""
from __future__ import annotations

from collections.abc import Callable

from vision.document.base import ReceiptExtraction


class FakeDocumentProvider:
    def __init__(self, extractions: list[ReceiptExtraction] | None = None,
                 responder: Callable[[bytes, str], ReceiptExtraction] | None = None) -> None:
        self._extractions = list(extractions or [])
        self._responder = responder
        self.calls: list[tuple[bytes, str]] = []

    def extract_receipt(self, file_bytes: bytes, *, mime_type: str) -> ReceiptExtraction:
        self.calls.append((file_bytes, mime_type))
        if self._responder is not None:
            return self._responder(file_bytes, mime_type)
        if self._extractions:
            return self._extractions.pop(0)
        return ReceiptExtraction()
