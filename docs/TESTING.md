# Testing

## Running the suite

```bash
pip install -e ".[dev]"   # or just: pip install pytest
python -m pytest -q
```

No network access, no API key, and no external binaries beyond `python3` itself are required — every
test that would otherwise need an LLM uses `core.llm.fake.FakeProvider` or
`core.llm.null.NullProvider`, every test that would otherwise need vision uses
`vision.ocr.fake.FakeOCRProvider`/`vision.document.fake.FakeDocumentProvider`, and the process-tool
tests only invoke `python3` (always present in this environment).

As of this writing: **171 tests, all passing**, covering every Phase 1 subsystem plus Phase 2 vision.

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
| `test_finance_receipt_import.py` | Receipt → transaction: happy path + category inference, missing/unparseable amount, missing/unparseable date fallback, currency fallback, dedup, vision-provider failure — all via `FakeDocumentProvider` |
| `test_finance_receipt_tool.py` | `finance_import_receipt` tool: sandboxed read, mime-type guessing/override, unsupported file type, extraction failure, duplicate reporting |
| `test_planner.py` | Rule-based intent recognition + failure, LLM planner validation/fallback (via `FakeProvider`) |
| `test_agent_loop.py` | Happy path, transient-failure-then-correction, permanent failure reporting FAILED (never a fabricated COMPLETE), unplannable request reporting BLOCKED, plan/step persistence |
| `test_scheduler.py` | Pure due-time computation for all three schedule kinds (including the spec's "every two weeks on Tuesday" example), `Scheduler.tick()` execution/skip/failure recording |
| `test_cli.py` | Subprocess smoke tests for every top-level command |

## Fixtures (`tests/conftest.py`)

`db` (in-memory, migrated), `sandbox` (scoped to `tmp_path`), `settings`, `event_bus`, `ctx` (a real
`ToolContext` with a real session row, so FK constraints hold), `registry`/`strict_registry` (every
registered tool, including `finance_import_receipt`, with a `PreApprovedGate` or `DenyAllGate`
respectively — note `finance_import_receipt` still needs a `provider=` override or
`ANTHROPIC_API_KEY` to actually run; tests that exercise it construct the tool directly with a
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
