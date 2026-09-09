# Testing

## Running the suite

```bash
pip install -e ".[dev]"   # or just: pip install pytest
python -m pytest -q
```

No network access and no API key are required — every test that would otherwise need an LLM uses
`core.llm.fake.FakeProvider`/`core.llm.null.NullProvider`, every test that would otherwise need
vision uses `vision.ocr.fake.FakeOCRProvider`/`vision.document.fake.FakeDocumentProvider`, and the
process-tool tests only invoke `python3` (always present in this environment). Document-generation
tests (`test_documents_*.py`) do exercise the real `python-docx`/`python-pptx`/`reportlab` libraries
(they're pure offline/deterministic — no network either way) but use `pytest.importorskip` so the
suite degrades to skipping them, rather than failing, if `kanna[documents]` isn't installed; it is
included in the `dev` extra, so `pip install -e ".[dev]"` runs the full suite.

`tests/test_fedora_agent.py` exercises the real `FedoraAgent` against a live X11 session — it needs
`DISPLAY` plus `xdotool`/`scrot`/`xclip`, none of which exist by default in most environments, so the
whole file is `pytest.mark.skipif`'d when `tools.computer.fedora.is_available()` is `False`. Locally,
set up a virtual display to run it for real:

```bash
apt-get install -y xvfb xdotool scrot xclip   # dnf install on Fedora
Xvfb :99 -screen 0 1280x800x24 &
DISPLAY=:99 python -m pytest -q tests/test_fedora_agent.py
```

CI does exactly this (see `.github/workflows/tests.yml`: installs the same four packages, runs the
whole suite under `xvfb-run`), so this file runs for real on every push rather than perpetually
skipping — see `docs/DEVICES.md` for what it actually validates and the real bug it caught.
`tests/test_computer_tools.py` covers the tool layer (schema, permissions, error translation) via
`tools.computer.fake.FakeComputerAgent` instead, so that coverage never depends on a display.

As of this writing: **257 tests when a display is available (245 + 12 skipped without one), all
passing**, covering every Phase 1 subsystem plus Phase 2 vision (receipts and statements), document
generation, and computer control.

## Layout

| File | Covers |
|---|---|
| `test_schema.py` | `Schema` validation (types, enum, bounds, pattern, nested object/array) + JSON-Schema export |
| `test_registry.py` | Tool registration, invocation pipeline (validate → permission → execute → validate → log), exception containment, output-schema enforcement |
| `test_tool_result.py` | `ToolResult` construction helpers |
| `test_permissions.py` | `PermissionPolicy` default decisions per level, rule overrides, both `ApprovalGate` implementations |
| `test_sandbox.py` | Path traversal, absolute-path escape, symlink escape, multi-root resolution |
| `test_db.py` | Migration application + idempotency, foreign-key enforcement, file-backed persistence across connections |
| `test_repositories.py` | Session/task/execution-log repository CRUD |
| `test_fs_tools.py` | Every filesystem tool's success and failure paths, sandbox rejection |
| `test_process_tool.py` | stdout/stderr/exit-code capture, executable allowlist enforcement, timeout, output truncation |
| `test_money.py` | Parsing (symbols, codes, thousands separators, embedded in a sentence), rounding, currency-mismatch errors |
| `test_finance_nlp.py` | The spec's own worked example + merchant/date/category-hint extraction, query period parsing |
| `test_finance_transactions.py` | Transaction creation, category inference, date-boundary filtering, monthly/category totals, multi-currency separation |
| `test_finance_budgets.py` | Budget status under/over budget, spending alerts, category exclusion |
| `test_finance_recurring.py` | Every recurrence frequency, explicit month-end rollover (incl. leap year), applying due recurring expenses |
| `test_finance_csv.py` | Import success/error-per-row, dedup, export round-trip, custom column mapping |
| `test_finance_dates.py` | Date-string normalization across every supported format, unrecognized-format error |
| `test_vision_ocr.py` | `FakeOCRProvider` scripted results/responder callback |
| `test_vision_document.py` | `FakeDocumentProvider` + `parse_receipt_json` (valid, all-null, wrong-typed fields, malformed line items) as a pure function |
| `test_vision_statement.py` | `FakeDocumentProvider` statement scripting + `parse_statement_json` (valid, all-null, wrong-typed fields, non-object rows skipped without losing valid ones, unrecognized `direction` rejected) as a pure function |
| `test_finance_receipt_import.py` | Receipt → transaction: happy path + category inference, missing/unparseable amount, missing/unparseable date fallback, currency fallback, dedup, vision-provider failure — all via `FakeDocumentProvider` |
| `test_finance_receipt_tool.py` | `finance_import_receipt` tool: sandboxed read, mime-type guessing/override, unsupported file type, extraction failure, duplicate reporting |
| `test_finance_statement_import.py` | Statement → transactions: debit-only import, credit/unclear-direction rows reported not imported, missing/unparseable amount, date fallback, currency fallback, dedup, no-transactions-found and provider-failure as document-level errors, multiple debits all imported |
| `test_finance_statement_tool.py` | `finance_import_statement` tool: partial-success reporting (unlike the all-or-nothing receipt tool), all-credits statement still a successful call with nothing created, sandboxing, mime-type override |
| `test_planner.py` | Rule-based intent recognition + failure, LLM planner validation/fallback (via `FakeProvider`) |
| `test_agent_loop.py` | Happy path, transient-failure-then-correction, permanent failure reporting FAILED (never a fabricated COMPLETE), unplannable request reporting BLOCKED, plan/step persistence |
| `test_scheduler.py` | Pure due-time computation for all three schedule kinds (including the spec's "every two weeks on Tuesday" example), `Scheduler.tick()` execution/skip/failure recording |
| `test_documents_model.py` | `build_document()` — the pure "args dict → `Document`" conversion (full content, empty-string-to-`None`, default heading level) |
| `test_documents_docx.py` | `render_docx` end-to-end, read back with `python-docx` to assert real structure (styles, table cells), not just file existence |
| `test_documents_pptx.py` | `render_pptx` end-to-end, incl. the table-only-section-must-keep-its-heading regression (see `docs/DOCUMENTS.md`) |
| `test_documents_pdf.py` | `render_pdf` end-to-end (valid `%PDF-` header, non-trivial size), incl. a regression test for XML-escaping special characters and for out-of-range heading levels |
| `test_documents_tools.py` | All three `document_generate_*` tools: creation, sandbox rejection, overwrite protection, directory-path rejection |
| `test_bootstrap.py` | `build_registry()` includes every subsystem's tools; `default_policy()`'s create-vs-overwrite rule, generalized to cover `fs_write_file` and all three `document_generate_*` tools |
| `test_computer_null.py` | Every `NullComputerAgent` method raises `CapabilityUnavailable` with a real reason |
| `test_fedora_agent.py` | The real `FedoraAgent` against a live X11 session (see above) — screenshot validity, a real click→type→Ctrl-D→read-the-file round trip, clipboard round trip, open/close application, honest failures with no display or a missing binary |
| `test_computer_selection.py` | `get_computer_agent()` picks `FedoraAgent` vs `NullComputerAgent` correctly |
| `test_computer_tools.py` | All 11 `computer_*` tools via `FakeComputerAgent`: argument passing, permission levels, `CapabilityUnavailable`/`ValueError` → `ToolResult.fail` translation, default-agent fallback |
| `test_cli.py` | Subprocess smoke tests for every top-level command |

## Fixtures (`tests/conftest.py`)

`db` (in-memory, migrated), `sandbox` (scoped to `tmp_path`), `settings`, `event_bus`, `ctx` (a real
`ToolContext` with a real session row, so FK constraints hold), `registry`/`strict_registry` (every
registered tool, including `finance_import_receipt`/`finance_import_statement`, with a
`PreApprovedGate` or `DenyAllGate` respectively — note those two still need a `provider=` override or
`ANTHROPIC_API_KEY` to actually run; tests that exercise them construct the tool directly with a
`FakeDocumentProvider` rather than going through this fixture).

## Principles for adding tests

- **Determinism first.** Finance and scheduler logic must never depend on wall-clock time implicitly
  — pass an explicit `now`/`as_of_date` everywhere it matters (see how every finance/scheduler test
  does this).
- **No network.** Anything LLM-shaped goes through `FakeProvider`.
- **Test the failure path, not just the happy path.** Every tool test file includes at least one
  rejected-input and one not-found/already-exists case; the agent-loop tests specifically assert that
  a permanently-failing step is reported as FAILED and never silently reported as success — that
  guarantee is the whole point of the verifier design and deserves its own explicit test.
- Run `python -m pytest -q` before considering any change complete; a change that breaks an existing
  test is not done until the test is fixed or the test's assumption is deliberately and explicitly
  updated.
