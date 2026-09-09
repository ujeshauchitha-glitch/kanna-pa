"""Receipt structure extraction backed by Claude's vision capability.

One multimodal call does both the transcription and the structuring —
the model reads the receipt and returns JSON, which is validated before
anything downstream trusts it (the same "validate the model's JSON
before using it" discipline `core.planner.llm_planner.LLMPlanner` uses
for plans). This is extraction, not computation: the model is reading a
printed total off the page, not computing one — `finance/analytics.py`
still does every aggregation over the amount that gets persisted. See
`docs/VISION.md` and `docs/FINANCE.md` for why that distinction matters.
"""
from __future__ import annotations

import json
import re

from vision._common import content_block, get_anthropic_client
from vision.document.base import LineItem, ReceiptExtraction

_INSTRUCTION = """Read this receipt and respond with ONLY a JSON object (no prose, no markdown \
fences) of the form:

{"merchant": "<store/vendor name>" or null,
 "date": "<the date printed on the receipt, in whatever format it appears>" or null,
 "currency": "<3-letter ISO currency code if you can determine it>" or null,
 "total_amount": "<the final total amount, as printed, digits and decimal point only>" or null,
 "line_items": [{"description": "<item>", "amount": "<amount as printed>" or null}, ...],
 "notes": "<anything uncertain or illegible you want the reader to know, or empty string>"}

Use null for any field you cannot confidently read. Do not guess a value you cannot see — an \
honest null is far better than an invented number. "total_amount" should be the final total the \
customer paid, not a subtotal, unless no total is printed."""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class AnthropicDocumentProvider:
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

    def extract_receipt(self, file_bytes: bytes, *, mime_type: str) -> ReceiptExtraction:
        client = self._get_client()
        block = content_block(file_bytes, mime_type)

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": [block, {"type": "text", "text": _INSTRUCTION}]}],
        )
        raw_text = "".join(b.text for b in response.content if b.type == "text")

        match = _JSON_BLOCK_RE.search(raw_text)
        if not match:
            return ReceiptExtraction(raw_text=raw_text, notes="model did not return parseable JSON")

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return ReceiptExtraction(raw_text=raw_text, notes="model returned malformed JSON")

        return parse_receipt_json(data, raw_text=raw_text)


def parse_receipt_json(data: dict, *, raw_text: str = "") -> ReceiptExtraction:
    """Validate and convert the model's JSON into a `ReceiptExtraction`.

    Tolerant of missing/null fields (a receipt extraction is inherently
    partial), strict about *types* when a field is present — a
    `total_amount` that isn't a string, for instance, is dropped with a
    note rather than trusted.
    """
    notes: list[str] = []
    if isinstance(data.get("notes"), str) and data["notes"]:
        notes.append(data["notes"])

    def _optional_str(key: str) -> str | None:
        value = data.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            notes.append(f"'{key}' was not a string in the model's response; ignored")
            return None
        return value or None

    merchant = _optional_str("merchant")
    date_text = _optional_str("date")
    currency_code = _optional_str("currency")
    total_amount_text = _optional_str("total_amount")

    line_items: list[LineItem] = []
    raw_items = data.get("line_items")
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict) or not isinstance(item.get("description"), str):
                continue
            amount = item.get("amount")
            line_items.append(LineItem(
                description=item["description"],
                amount_text=amount if isinstance(amount, str) else None,
            ))
    elif raw_items is not None:
        notes.append("'line_items' was not a list in the model's response; ignored")

    return ReceiptExtraction(
        merchant=merchant, date_text=date_text, currency_code=currency_code,
        total_amount_text=total_amount_text, line_items=line_items, raw_text=raw_text,
        notes="; ".join(notes),
    )
