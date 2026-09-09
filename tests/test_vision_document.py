from __future__ import annotations

from vision.document.anthropic_document import parse_receipt_json
from vision.document.base import LineItem, ReceiptExtraction
from vision.document.fake import FakeDocumentProvider


def test_fake_document_provider_returns_scripted_extraction():
    extraction = ReceiptExtraction(merchant="Cafe Coffee Day", total_amount_text="340.00")
    provider = FakeDocumentProvider(extractions=[extraction])
    result = provider.extract_receipt(b"bytes", mime_type="image/jpeg")
    assert result.merchant == "Cafe Coffee Day"
    assert result.total_amount_text == "340.00"
    assert provider.calls == [(b"bytes", "image/jpeg")]


def test_fake_document_provider_default_is_empty_extraction():
    provider = FakeDocumentProvider()
    result = provider.extract_receipt(b"x", mime_type="image/png")
    assert result == ReceiptExtraction()


# -- parse_receipt_json: the pure JSON -> ReceiptExtraction conversion the
# real AnthropicDocumentProvider uses after parsing the model's response. --

def test_parse_receipt_json_full():
    data = {
        "merchant": "Starbucks", "date": "2024-03-15", "currency": "INR",
        "total_amount": "450.00",
        "line_items": [{"description": "Latte", "amount": "250.00"},
                        {"description": "Muffin", "amount": "200.00"}],
        "notes": "",
    }
    result = parse_receipt_json(data, raw_text="raw ocr text")
    assert result.merchant == "Starbucks"
    assert result.date_text == "2024-03-15"
    assert result.currency_code == "INR"
    assert result.total_amount_text == "450.00"
    assert result.line_items == [LineItem("Latte", "250.00"), LineItem("Muffin", "200.00")]
    assert result.raw_text == "raw ocr text"
    assert result.notes == ""


def test_parse_receipt_json_all_null():
    data = {"merchant": None, "date": None, "currency": None, "total_amount": None,
            "line_items": [], "notes": "receipt too blurry to read"}
    result = parse_receipt_json(data)
    assert result.merchant is None
    assert result.total_amount_text is None
    assert result.notes == "receipt too blurry to read"


def test_parse_receipt_json_tolerates_wrong_types():
    data = {"merchant": 123, "total_amount": ["not", "a", "string"], "line_items": "not a list"}
    result = parse_receipt_json(data)
    assert result.merchant is None
    assert result.total_amount_text is None
    assert result.line_items == []
    assert "merchant" in result.notes
    assert "total_amount" in result.notes
    assert "line_items" in result.notes


def test_parse_receipt_json_skips_malformed_line_items():
    data = {"line_items": [{"description": "ok item", "amount": "5"}, {"no_description": True}, "junk"]}
    result = parse_receipt_json(data)
    assert result.line_items == [LineItem("ok item", "5")]
