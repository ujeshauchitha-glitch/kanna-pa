# Vision

## Status: one real provider, Anthropic-backed

Phase 2 adds `vision/`, with four capabilities implemented — OCR (plain text transcription), receipt
structure extraction, multi-transaction statement extraction, and generic document structure
extraction — all backed by Claude's multimodal vision, and all following the same "protocol +
swappable implementation + fake for tests" pattern as `core/llm/`.

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
                        StatementExtraction, StatementTransaction, DocumentStructure,
                        StructureSection, StructureTable
    anthropic_document.py  Real implementation — one Claude call reads + structures a receipt, a
                        (possibly multi-page) statement, or an arbitrary document
    fake.py             FakeDocumentProvider for tests
  tools.py              vision_extract_structure — the one vision capability with no finance-shaped
                        consumer, so it's registered directly rather than wrapped by another
                        subsystem's tool (see below)
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

## Document extraction (`vision/document/`) — receipts, statements, and generic structure

`DocumentProvider.extract_receipt(file_bytes, *, mime_type) -> ReceiptExtraction`,
`.extract_statement(file_bytes, *, mime_type) -> StatementExtraction`, and
`.extract_structure(file_bytes, *, mime_type) -> DocumentStructure` (all three accept a multi-page
PDF — Claude's document content block supports that directly, no page-splitting needed). The receipt
and statement extractions deliberately return **raw, unparsed strings** — `date_text`,
`currency_code`/`account_currency`, `total_amount_text`/`amount_text` — not a `date` or a `Money`.
Converting "340.50" or "15/03/2024" into an exact value is `finance`'s job (`finance.money.
Money.parse`, `finance.dates.normalize_date`), not vision's. This keeps the dependency direction
one-way: `finance` depends on `vision`, never the other way around.

`AnthropicDocumentProvider` asks Claude for one JSON object per call — for a receipt: merchant, date,
currency, total_amount, line_items, notes; for a statement: account_currency and a list of
transaction rows (date, description, amount, `direction`: `"debit"`/`"credit"`/`null`); for generic
structure: an optional title and a list of sections (heading, level, paragraphs, bullets, an optional
table) — explicitly instructed to use `null` rather than guess anything it can't confidently read,
told to leave `direction` `null` (never guess) when it genuinely can't tell debit from credit, and
told to preserve the document's own wording rather than summarize or paraphrase. All three responses
are parsed defensively: `parse_receipt_json()`/`parse_statement_json()`/`parse_structure_json()` are
pure functions, independently unit-tested, that tolerate missing/null fields but drop (with a note,
never silently) any field that comes back the wrong type, and skip (with a note) any row/section
that isn't even a JSON object rather than losing every other real one over a single bad entry.

### Generic structure extraction (`extract_structure`)

Unlike the receipt/statement shapes (a fixed transaction-row schema), `DocumentStructure` generalizes
to arbitrary content: a report, an article, a PDF assignment — sections with headings, body
paragraphs, bullet lists, and tables, in whatever structure the source document actually has (its own
headings if it has them, natural topic breaks if it doesn't). `StructureSection` is deliberately
shaped like `documents.model.Section` (heading/level/paragraphs/bullets/table) so a caller that wants
to *re-render* an extracted document (read a PDF, write it back out as a DOCX, say) can map one to the
other directly — but `vision` doesn't import `documents` itself; that mapping, if a caller wants it,
lives on their side, the same one-way-dependency discipline `finance` already follows.

There's no finance-shaped consumer for this one (a document's structure isn't a spending transaction),
so it's exposed directly as its own tool rather than wrapped inside another subsystem's: see
`vision_extract_structure` below.

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

## Generic structure extraction tool (`vision/tools.py`)

`vision_extract_structure` (LOW permission — read-only) reads a sandboxed path, guesses (or accepts an
override for) its mime type the same way `finance_import_receipt`/`finance_import_statement` do, and
returns the document's title, sections (heading/level/paragraphs/bullets), and any tables — each
section's `table` key is present only when that section actually has one, never `null` (an
object-typed schema field can't validate a `null` value, so it's omitted instead — see
`vision/tools.py::_section_data`). Fails cleanly with `vision_unavailable`
(`ANTHROPIC_API_KEY`/`anthropic` package missing) or `structure_extraction_failed` (any other provider
error — auth, network) rather than crashing. No CLI subcommand yet (unlike `finance import-receipt`) —
registry/agent-loop path only.

## Testing

Every test uses `FakeOCRProvider`/`FakeDocumentProvider` (`vision/ocr/fake.py`,
`vision/document/fake.py`) — no test in the suite touches the network or requires
`ANTHROPIC_API_KEY`. `parse_receipt_json()`/`parse_statement_json()`/`parse_structure_json()` are
tested directly as pure functions (valid data, all-null data, wrong-typed fields, malformed
rows/line-items/sections/tables) independent of any actual API call. See `tests/test_vision_ocr.py`,
`tests/test_vision_document.py`, `tests/test_vision_statement.py`, `tests/test_vision_structure.py`,
`tests/test_vision_tools.py`, `tests/test_finance_dates.py`, `tests/test_finance_receipt_import.py`,
`tests/test_finance_receipt_tool.py`, `tests/test_finance_statement_import.py`,
`tests/test_finance_statement_tool.py`.

Manually verified end-to-end that the failure path is honest: with no `ANTHROPIC_API_KEY` set,
`kanna finance import-receipt <path>` and `kanna finance import-statement <path>` both fail clearly
(`VisionUnavailable`, surfaced as a normal tool error) rather than crashing or fabricating a
transaction; `vision_extract_structure` follows the identical pattern through the registry.

## Known limitations

- No offline/local OCR provider (would need `tesseract` or similar, not installed in this
  environment).
- Generic structure extraction doesn't understand a document's *semantics* beyond its visual
  structure — it won't distinguish "an assignment's instructions" from "an assignment's questions"
  unless the source document's own headings already do; that's a future assignment-reading workflow's
  job, built on top of this (see `docs/ROADMAP.md`).
- Statement import only imports debit rows — no income/credit tracking (see
  `finance/imports/statement.py`'s docstring for the reasoning and where that boundary would move).
- No image editing/generation, no screenshot-understanding tied to `tools/computer/` yet.
- `AnthropicDocumentProvider` makes one API call per document; no batching, no caching of a
  re-imported identical file (though the resulting transactions are still deduped by content hash per
  row — you'd just pay for a redundant API call, not get duplicate rows).
