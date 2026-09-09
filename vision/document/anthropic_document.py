"""Receipt and statement structure extraction backed by Claude's vision capability.

One multimodal call does both the transcription and the structuring —
the model reads the document and returns JSON, which is validated before
anything downstream trusts it (the same "validate the model's JSON
before using it" discipline `core.planner.llm_planner.LLMPlanner` uses
for plans). This is extraction, not computation: the model is reading
printed numbers off the page, never computing one — `finance/
analytics.py` still does every aggregation over whatever amount gets
persisted. See `docs/VISION.md` and `docs/FINANCE.md` for why that
distinction matters.
"""
from __future__ import annotations

import json
import re

from vision._common import content_block, get_anthropic_client
from vision.document.base import LineItem, ReceiptExtraction, StatementExtraction, StatementTransaction

_RECEIPT_INSTRUCTION = """Read this receipt and respond with ONLY a JSON object (no prose, no markdown \
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

_STATEMENT_INSTRUCTION = """Read this bank/card statement (it may span multiple pages) and respond \
with ONLY a JSON object (no prose, no markdown fences) of the form:

{"account_currency": "<3-letter ISO currency code if determinable>" or null,
 "transactions": [
   {"date": "<the date printed for this transaction, as printed>" or null,
    "description": "<the transaction description/merchant/memo as printed>" or null,
    "amount": "<the amount for this transaction, digits and decimal point only, unsigned>" or null,
    "direction": "debit" or "credit" or null},
   ...
 ],
 "notes": "<anything uncertain, illegible, or that seems like a running balance rather than a \
transaction — call it out here rather than guessing, or empty string>"}

List every individual transaction row you can find, in the order they appear. Use "debit" for money \
leaving the account (a purchase, a payment, a withdrawal) and "credit" for money entering it (a \
deposit, a refund, incoming transfer) — read this from the statement's own column headers or \
+/- signs; if a row's direction genuinely can't be determined, use null rather than guessing. Do not \
include summary/subtotal/running-balance rows as transactions. Use null for any field on a given \
row you cannot confidently read — an honest null is far better than an invented value."""

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

    def _complete_json(self, file_bytes: bytes, mime_type: str, instruction: str,
                        max_tokens: int | None = None) -> tuple[dict | None, str, str]:
        """Send one document+instruction call, return (parsed JSON or None, raw text, error note)."""
        client = self._get_client()
        block = content_block(file_bytes, mime_type)

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            messages=[{"role": "user", "content": [block, {"type": "text", "text": instruction}]}],
        )
        raw_text = "".join(b.text for b in response.content if b.type == "text")

        match = _JSON_BLOCK_RE.search(raw_text)
        if not match:
            return None, raw_text, "model did not return parseable JSON"
        try:
            return json.loads(match.group(0)), raw_text, ""
        except json.JSONDecodeError:
            return None, raw_text, "model returned malformed JSON"

    def extract_receipt(self, file_bytes: bytes, *, mime_type: str) -> ReceiptExtraction:
        data, raw_text, error = self._complete_json(file_bytes, mime_type, _RECEIPT_INSTRUCTION)
        if data is None:
            return ReceiptExtraction(raw_text=raw_text, notes=error)
        return parse_receipt_json(data, raw_text=raw_text)

    def extract_statement(self, file_bytes: bytes, *, mime_type: str) -> StatementExtraction:
        # A multi-page statement can list many rows — give the model more
        # room than the receipt/single-transaction case.
        data, raw_text, error = self._complete_json(
            file_bytes, mime_type, _STATEMENT_INSTRUCTION, max_tokens=max(self.max_tokens, 8192)
        )
        if data is None:
            return StatementExtraction(raw_text=raw_text, notes=error)
        return parse_statement_json(data, raw_text=raw_text)


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

    def _optional_str(source: dict, key: str) -> str | None:
        value = source.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            notes.append(f"'{key}' was not a string in the model's response; ignored")
            return None
        return value or None

    merchant = _optional_str(data, "merchant")
    date_text = _optional_str(data, "date")
    currency_code = _optional_str(data, "currency")
    total_amount_text = _optional_str(data, "total_amount")

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


_VALID_DIRECTIONS = {"debit", "credit"}


def parse_statement_json(data: dict, *, raw_text: str = "") -> StatementExtraction:
    """Validate and convert the model's JSON into a `StatementExtraction`.

    Same tolerance rules as `parse_receipt_json`: missing/null is fine,
    wrong-typed is dropped with a note. A malformed individual
    transaction row is skipped (noted) rather than discarding the whole
    statement — one bad row shouldn't cost every other real transaction.
    """
    notes: list[str] = []
    if isinstance(data.get("notes"), str) and data["notes"]:
        notes.append(data["notes"])

    account_currency = data.get("account_currency")
    if account_currency is not None and not isinstance(account_currency, str):
        notes.append("'account_currency' was not a string in the model's response; ignored")
        account_currency = None
    elif account_currency == "":
        account_currency = None

    transactions: list[StatementTransaction] = []
    raw_transactions = data.get("transactions")
    if isinstance(raw_transactions, list):
        for i, row in enumerate(raw_transactions):
            if not isinstance(row, dict):
                notes.append(f"transaction {i}: not an object; skipped")
                continue

            def _field(key: str) -> str | None:
                value = row.get(key)
                return value if isinstance(value, str) and value else None

            direction = _field("direction")
            if direction is not None and direction not in _VALID_DIRECTIONS:
                notes.append(f"transaction {i}: unrecognized direction {direction!r}; ignored")
                direction = None

            transactions.append(StatementTransaction(
                date_text=_field("date"), description=_field("description"),
                amount_text=_field("amount"), direction=direction,
            ))
    elif raw_transactions is not None:
        notes.append("'transactions' was not a list in the model's response; ignored")

    return StatementExtraction(
        account_currency=account_currency, transactions=transactions, raw_text=raw_text,
        notes="; ".join(notes),
    )
