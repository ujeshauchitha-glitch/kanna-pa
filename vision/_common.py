"""Shared helpers for Anthropic-backed vision providers.

Both `vision/ocr/anthropic_ocr.py` and `vision/document/anthropic_document.py`
talk to the same API the same way — lazy client construction, base64
encoding, and building the right content-block type for an image vs. a
PDF. Factored here so neither provider duplicates that boilerplate.
"""
from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from core.errors import VisionUnavailable

_SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
_SUPPORTED_DOCUMENT_TYPES = {"application/pdf"}
SUPPORTED_MIME_TYPES = _SUPPORTED_IMAGE_TYPES | _SUPPORTED_DOCUMENT_TYPES


def guess_mime_type(path: str | Path) -> str:
    """Guess a supported mime type from a file extension.

    Raises `ValueError` for anything not in `SUPPORTED_MIME_TYPES` — callers
    should surface that as a clear tool failure rather than guessing further.
    """
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"unsupported or unrecognized file type for {path!r} (guessed {mime_type!r}); "
            f"supported types: {sorted(SUPPORTED_MIME_TYPES)}"
        )
    return mime_type


def get_anthropic_client(*, api_key: str | None = None):
    """Lazily construct an `anthropic.Anthropic` client.

    Raises `VisionUnavailable` — never falls back silently — when the
    package isn't installed or no API key is configured, mirroring
    `core.llm.anthropic_provider.AnthropicProvider._get_client`.
    """
    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise VisionUnavailable(
            "ANTHROPIC_API_KEY is not set; vision features need it, or configure a "
            "different vision provider"
        )
    try:
        import anthropic
    except ImportError as exc:
        raise VisionUnavailable(
            "the 'anthropic' package is not installed; run `pip install kanna[vision]`"
        ) from exc

    return anthropic.Anthropic(api_key=resolved_key)


def content_block(file_bytes: bytes, mime_type: str) -> dict:
    """Build the Messages API content block for an image or a PDF document."""
    encoded = base64.standard_b64encode(file_bytes).decode("ascii")
    if mime_type in _SUPPORTED_IMAGE_TYPES:
        return {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": encoded}}
    if mime_type in _SUPPORTED_DOCUMENT_TYPES:
        return {"type": "document", "source": {"type": "base64", "media_type": mime_type, "data": encoded}}
    raise ValueError(f"unsupported mime type: {mime_type!r}; supported: {sorted(SUPPORTED_MIME_TYPES)}")
