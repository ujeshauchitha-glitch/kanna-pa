"""OCR backed by Claude's vision capability.

There's no local OCR engine bundled (this environment has no `tesseract`
binary — see `docs/VISION.md`), so the one real provider Phase 2 ships is
this one: it asks Claude to transcribe an image or PDF verbatim. Requires
`ANTHROPIC_API_KEY`; raises `VisionUnavailable` rather than silently
returning empty text when it isn't configured.
"""
from __future__ import annotations

from vision._common import content_block, get_anthropic_client
from vision.ocr.base import OCRResult

_TRANSCRIBE_INSTRUCTION = (
    "Transcribe every piece of text visible in this document, verbatim and in reading order. "
    "Output only the transcribed text — no commentary, no markdown formatting, no summary."
)


class AnthropicOCRProvider:
    def __init__(self, *, model: str = "claude-sonnet-5", max_tokens: int = 4096,
                 api_key: str | None = None) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = get_anthropic_client(api_key=self._api_key)
        return self._client

    def extract_text(self, file_bytes: bytes, *, mime_type: str) -> OCRResult:
        client = self._get_client()
        block = content_block(file_bytes, mime_type)

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": [block, {"type": "text", "text": _TRANSCRIBE_INSTRUCTION}]}],
        )

        text = "".join(b.text for b in response.content if b.type == "text")
        return OCRResult(text=text, confidence=None, metadata={"model": self.model})
