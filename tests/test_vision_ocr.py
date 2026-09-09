from __future__ import annotations

from vision.ocr.base import OCRResult
from vision.ocr.fake import FakeOCRProvider


def test_fake_ocr_returns_scripted_result():
    provider = FakeOCRProvider(results=[OCRResult(text="hello world")])
    result = provider.extract_text(b"fake-bytes", mime_type="image/png")
    assert result.text == "hello world"
    assert result.confidence is None
    assert provider.calls == [(b"fake-bytes", "image/png")]


def test_fake_ocr_default_result_is_empty():
    provider = FakeOCRProvider()
    result = provider.extract_text(b"x", mime_type="image/png")
    assert result.text == ""


def test_fake_ocr_responder_callback():
    provider = FakeOCRProvider(responder=lambda data, mime: OCRResult(text=f"{mime}:{len(data)}"))
    result = provider.extract_text(b"abc", mime_type="application/pdf")
    assert result.text == "application/pdf:3"
