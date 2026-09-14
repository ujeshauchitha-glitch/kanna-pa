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
from vision.document.base import (
    DocumentStructure, LineItem, ReceiptExtraction, StatementExtraction, StatementTransaction,
    StructureSection, StructureTable,
)

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

_STRUCTURE_INSTRUCTION = """Read this document (it may span multiple pages) and respond with ONLY a \
JSON object (no prose, no markdown fences) of the form:

{"title": "<the document's title, as printed>" or null,
 "sections": [
   {"heading": "<this section's heading, as printed>" or null,
    "level": <1 for a top-level heading, 2 for a subsection, 3 for a sub-subsection, ...>,
    "paragraphs": ["<paragraph text>", ...],
    "bullets": ["<bullet/list item text>", ...],
    "table": {"headers": ["<column>", ...], "rows": [["<cell>", ...], ...]} or null},
   ...
 ],
 "notes": "<anything uncertain, illegible, or structurally ambiguous — call it out here rather than \
guessing, or empty string>"}

Break the document into sections the way a reader would — by its own headings if it has them, or by \
natural topic breaks if it doesn't (in which case use "heading": null for that section and level 1). \
Preserve the document's own wording; do not summarize, paraphrase, or add content that isn't printed. \
A section that lists items should have those items in "bullets" if unnumbered/unordered, and in \
"paragraphs" otherwise; use "table" only for content actually laid out as rows and columns. Use null \
for any field you cannot confidently read — an honest null is far better than an invented value."""

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

    def extract_structure(self, file_bytes: bytes, *, mime_type: str) -> DocumentStructure:
        # A multi-page document can have a lot of sections/paragraphs —
        # same reasoning as the statement case for extra headroom.
        data, raw_text, error = self._complete_json(
            file_bytes, mime_type, _STRUCTURE_INSTRUCTION, max_tokens=max(self.max_tokens, 8192)
        )
        if data is None:
            return DocumentStructure(raw_text=raw_text, notes=error)
        return parse_structure_json(data, raw_text=raw_text)


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


def _parse_structure_table(raw: object, notes: list[str], index: int) -> StructureTable | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        notes.append(f"section {index}: 'table' was not an object in the model's response; ignored")
        return None

    headers = raw.get("headers")
    if not isinstance(headers, list) or not all(isinstance(h, str) for h in headers):
        notes.append(f"section {index}: table 'headers' was not a list of strings; ignored")
        headers = []

    rows: list[list[str]] = []
    raw_rows = raw.get("rows")
    if isinstance(raw_rows, list):
        for row in raw_rows:
            if isinstance(row, list) and all(isinstance(cell, str) for cell in row):
                rows.append(row)
            else:
                notes.append(f"section {index}: a table row was not a list of strings; skipped")
    elif raw_rows is not None:
        notes.append(f"section {index}: table 'rows' was not a list; ignored")

    return StructureTable(headers=headers, rows=rows)


def parse_structure_json(data: dict, *, raw_text: str = "") -> DocumentStructure:
    """Validate and convert the model's JSON into a `DocumentStructure`.

    Same tolerance rules as `parse_receipt_json`/`parse_statement_json`:
    missing/null is fine, wrong-typed is dropped with a note. A malformed
    individual section is skipped (noted) rather than discarding the
    whole document.
    """
    notes: list[str] = []
    if isinstance(data.get("notes"), str) and data["notes"]:
        notes.append(data["notes"])

    title = data.get("title")
    if title is not None and not isinstance(title, str):
        notes.append("'title' was not a string in the model's response; ignored")
        title = None
    elif title == "":
        title = None

    sections: list[StructureSection] = []
    raw_sections = data.get("sections")
    if isinstance(raw_sections, list):
        for i, raw in enumerate(raw_sections):
            if not isinstance(raw, dict):
                notes.append(f"section {i}: not an object; skipped")
                continue

            heading = raw.get("heading")
            if heading is not None and not isinstance(heading, str):
                notes.append(f"section {i}: 'heading' was not a string; ignored")
                heading = None
            elif heading == "":
                heading = None

            level = raw.get("level")
            level = level if isinstance(level, int) and level > 0 else 1

            paragraphs = raw.get("paragraphs")
            if isinstance(paragraphs, list) and all(isinstance(p, str) for p in paragraphs):
                paragraphs = list(paragraphs)
            else:
                if paragraphs is not None:
                    notes.append(f"section {i}: 'paragraphs' was not a list of strings; ignored")
                paragraphs = []

            bullets = raw.get("bullets")
            if isinstance(bullets, list) and all(isinstance(b, str) for b in bullets):
                bullets = list(bullets)
            else:
                if bullets is not None:
                    notes.append(f"section {i}: 'bullets' was not a list of strings; ignored")
                bullets = []

            sections.append(StructureSection(
                heading=heading, level=level, paragraphs=paragraphs, bullets=bullets,
                table=_parse_structure_table(raw.get("table"), notes, i),
            ))
    elif raw_sections is not None:
        notes.append("'sections' was not a list in the model's response; ignored")

    return DocumentStructure(title=title, sections=sections, raw_text=raw_text, notes="; ".join(notes))
