# Vision

## Status: one real provider, Anthropic-backed

Phase 2 adds `vision/`, with three capabilities implemented — OCR (plain text transcription), receipt
structure extraction, and multi-transaction statement extraction — all backed by Claude's multimodal
vision, and all following the same "protocol + swappable implementation + fake for tests" pattern as
`core/llm/`.

This environment has no local OCR engine (`tesseract` is not installed — see the environment note in
`KANNA_SPEC.md`), so there is no offline/local provider in Phase 2. That's an environment fact, not a
design choice: `vision/ocr/base.py` and `vision/document/base.py` are plain `Protocol`s, so a local
provider (e.g. `pytesseract`-backed) can be added later without touching any caller.

## Modules

```
vision/
  _common.py           Shared Anthropic-client + content-block helpers used by both providers below
  ocr/
    base.py             OCRProvider protocol, OCRResult
    anthropic_ocr.py    Real implementation — Claude transcribes an image/PDF verbatim
    fake.py             FakeOCRProvider for tests
  document/
    base.py             DocumentProvider protocol, ReceiptExtraction, LineItem,
                        StatementExtraction, StatementTransaction
    anthropic_document.py  Real implementation — one Claude call reads + structures a receipt or a
                        (possibly multi-page) statement
    fake.py             FakeDocumentProvider for tests
```

`vision/image/` (general image description/editing) and `vision/screen/` (screenshot understanding,
tied to `tools/computer/`) are not built yet — see `docs/ROADMAP.md`. They aren't scaffolded as empty
directories, per the same "no empty scaffolding" rule Phase 1 followed.

## OCR (`vision/ocr/`)

`OCRProvider.extract_text(file_bytes, *, mime_type) -> OCRResult(text, confidence, metadata)`.
`AnthropicOCRProvider` sends the image/PDF as a content block with an instruction to transcribe
verbatim, in reading order, nothing else. `confidence` is always `None` — Claude doesn't return a
calibrated confidence score, and fabricating one would violate the same "never invent a number"
principle the finance subsystem holds itself to. Requires `ANTHROPIC_API_KEY`; raises
`VisionUnavailable` (not a silent empty result) when it's missing.

## Document extraction (`vision/document/`) — receipts and statements

`DocumentProvider.extract_receipt(file_bytes, *, mime_type) -> ReceiptExtraction` and
`.extract_statement(file_bytes, *, mime_type) -> StatementExtraction` (the latter accepts a
multi-page PDF — Claude's document content block supports that directly, no page-splitting needed).
Both deliberately return **raw, unparsed strings** — `date_text`, `currency_code`/`account_currency`,
`total_amount_text`/`amount_text` — not a `date` or a `Money`. Converting "340.50" or "15/03/2024"
into an exact value is `finance`'s job (`finance.money.Money.parse`, `finance.dates.normalize_date`),
not vision's. This keeps the dependency direction one-way: `finance` depends on `vision`, never the
other way around.

`AnthropicDocumentProvider` asks Claude for one JSON object per call — for a receipt: merchant, date,
currency, total_amount, line_items, notes; for a statement: account_currency and a list of
transaction rows (date, description, amount, `direction`: `"debit"`/`"credit"`/`null`) — explicitly
instructed to use `null` rather than guess anything it can't confidently read, and told to leave
`direction` `null` (never guess) when it genuinely can't tell debit from credit. Both responses are
parsed defensively: `parse_receipt_json()`/`parse_statement_json()` are pure functions, independently
unit-tested, that tolerate missing/null fields but drop (with a note, never silently) any field that
comes back the wrong type, and skip (with a note) any transaction row that isn't even a JSON object
rather than losing every other real row in the statement over one bad one.

### This is extraction, not computation

`docs/FINANCE.md`'s rule — the LLM never computes a financial total — still holds. Reading the total
printed on a receipt is **extraction** (the same category of operation as `finance/nlp.py` regex-
parsing an amount out of a sentence), not aggregation. `finance/analytics.py` remains the only place
that sums, and it sums over whatever amount ends up persisted, regardless of whether that amount was
typed, said, or photographed. What vision adds is *one more way to get a single, already-final number
into the system* — it does not touch how monthly totals, budget status, or any other calculation is
produced.

## Receipt import (`finance/imports/receipt.py`)

The concrete, tested consumer that proves the vision subsystem out — the `ReceiptImporter` interface
`finance/imports/interfaces.py` defined in Phase 1 as "not yet implemented" now has a real
implementation:

```
receipt image/PDF bytes
  → DocumentProvider.extract_receipt()          (vision — reads the page)
  → Money.parse(total_amount_text, currency)     (finance — exact parsing, same as NL entry)
  → normalize_date(date_text) or today           (finance — same date handling as CSV import)
  → category via existing keyword rules           (finance — same inference as NL/CSV entry)
  → Transaction, deduped by content_hash          (finance — same repository, same dedup as CSV)
```

A receipt with no readable total is reported as an error and **creates nothing** — never a
zero-amount transaction. A missing or unparseable date falls back to today's date with a note
attached to the transaction (surfaced, not hidden). Exposed as `finance_import_receipt` (a LOW-
permission tool, same risk class as `finance_add_transaction`) and `kanna finance import-receipt
<path>`.

## Statement import (`finance/imports/statement.py`)

The `StatementImporter` interface's real implementation — same pipeline as receipt import, but over
every row `extract_statement()` returns instead of one document-level total:

```
statement image/PDF bytes (possibly multi-page)
  → DocumentProvider.extract_statement()                (vision — reads every row on every page)
  → for each row: Money.parse(amount_text, currency)      (finance — same exact parsing)
  → normalize_date(date_text) or today                    (finance — same date handling)
  → category via existing keyword rules                    (finance — same inference)
  → Transaction, deduped by content_hash                   (finance — same dedup as CSV/receipt)
```

**Only debit rows become transactions.** Kanna's finance subsystem tracks *spending* — every other
entry path (NL, CSV, receipt) records a purchase, and `Transaction` has no signed-amount or income/
expense field. A statement mixes debits (spend — fits the model) with credits (deposits, refunds,
incoming transfers — money *in*, which doesn't). Rather than invent a sign convention or silently drop
credits, each credit row — and each row whose direction the model couldn't determine — is reported in
the result as an explained skip: visible, never silently lost, never misrepresented as a purchase. A
statement that's all credits (a paycheck-only account, say) is still a *successful* read that simply
imports nothing — that's not a tool failure. Exposed as `finance_import_statement` (LOW permission)
and `kanna finance import-statement <path>`.

## Testing

Every test uses `FakeOCRProvider`/`FakeDocumentProvider` (`vision/ocr/fake.py`,
`vision/document/fake.py`) — no test in the suite touches the network or requires
`ANTHROPIC_API_KEY`. `parse_receipt_json()`/`parse_statement_json()` are tested directly as pure
functions (valid data, all-null data, wrong-typed fields, malformed rows/line items) independent of
any actual API call. See `tests/test_vision_ocr.py`, `tests/test_vision_document.py`,
`tests/test_vision_statement.py`, `tests/test_finance_dates.py`, `tests/test_finance_receipt_import.py`,
`tests/test_finance_receipt_tool.py`, `tests/test_finance_statement_import.py`,
`tests/test_finance_statement_tool.py`.

Manually verified end-to-end that the failure path is honest: with no `ANTHROPIC_API_KEY` set,
`kanna finance import-receipt <path>` and `kanna finance import-statement <path>` both fail clearly
(`VisionUnavailable`, surfaced as a normal tool error) rather than crashing or fabricating a
transaction.

## Known limitations

- No offline/local OCR provider (would need `tesseract` or similar, not installed in this
  environment).
- Receipt and statement extraction only — no *generic* document structure extraction yet (e.g. for
  reading a PDF assignment's questions/instructions, planned in `docs/ROADMAP.md`; a statement's
  fixed transaction-row shape doesn't generalize to arbitrary documents).
- Statement import only imports debit rows — no income/credit tracking (see
  `finance/imports/statement.py`'s docstring for the reasoning and where that boundary would move).
- No image editing/generation, no screenshot-understanding tied to `tools/computer/` yet.
- `AnthropicDocumentProvider` makes one API call per document; no batching, no caching of a
  re-imported identical file (though the resulting transactions are still deduped by content hash per
  row — you'd just pay for a redundant API call, not get duplicate rows).
