"""The OCR provider interface.

Deliberately narrow: given image or PDF bytes, return the text on the
page. `vision/document/` builds structured extraction (receipts, etc.)
on top of the same kind of provider but is implemented independently
rather than layered on this, since a single multimodal call can do both
transcription and structuring at once — see
`vision/document/anthropic_document.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class OCRResult:
    text: str
    # 0-1 if the provider reports a calibrated confidence, else None —
    # never fabricated. Claude's vision output doesn't include one, so
    # AnthropicOCRProvider always leaves this None.
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OCRProvider(Protocol):
    def extract_text(self, file_bytes: bytes, *, mime_type: str) -> OCRResult:
        ...
