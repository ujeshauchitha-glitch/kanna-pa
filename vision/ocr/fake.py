"""A scripted OCR provider for tests — no network, no test ever imports anthropic_ocr.py."""
from __future__ import annotations

from collections.abc import Callable

from vision.ocr.base import OCRResult


class FakeOCRProvider:
    def __init__(self, results: list[OCRResult] | None = None,
                 responder: Callable[[bytes, str], OCRResult] | None = None) -> None:
        self._results = list(results or [])
        self._responder = responder
        self.calls: list[tuple[bytes, str]] = []

    def extract_text(self, file_bytes: bytes, *, mime_type: str) -> OCRResult:
        self.calls.append((file_bytes, mime_type))
        if self._responder is not None:
            return self._responder(file_bytes, mime_type)
        if self._results:
            return self._results.pop(0)
        return OCRResult(text="")
