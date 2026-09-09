# Finance

## The one rule

**The LLM never computes a financial number.** The path from a sentence to an answer is always:

```
natural language
  → finance.nlp (deterministic regex parsing — no model call)
  → TransactionDraft / FinanceQuery (structured, validated)
  → finance.repository (SQL read/write)
  → finance.analytics (SQL aggregation + Decimal arithmetic — the only place a total is computed)
  → a plain-language sentence built from that already-computed number
```

An `LLMPlanner` may produce a `finance_add_transaction`/`finance_query` tool call for a phrasing the
regex parser doesn't cover, but it still only ever supplies the input **text** — the tool then runs
the identical deterministic pipeline. There is no code path where a model's own arithmetic becomes a
transaction amount or a reported total.

## Money representation

`finance.money.Money` stores every amount as an **integer in the currency's smallest unit**
(`amount_minor`) — paise for INR, cents for USD, no minor unit at all for JPY — never a float.
Parsing from text goes through `Decimal` and is quantized exactly once, at the boundary
(`Money.from_decimal`, `ROUND_HALF_UP`); every operation after that is plain integer arithmetic.
Combining two different currencies raises `CurrencyMismatch` rather than silently converting — Kanna
never invents an exchange rate. `Money.parse()` reads `₹`/`$`/`€`/`£`/`¥` symbols, a leading ISO code
("INR 1,234"), or falls back to the caller's default currency.

## Domain model

- **Transaction**: id, amount_minor, currency, occurred_at (date), category_id, subcategory,
  merchant, description, payment_method, source, notes, content_hash (for import dedup), created_at,
  updated_at.
- **Category**: id, name, parent_category (self-referential, for subcategories).
- **Budget**: id, category_id (nullable = overall), amount_minor, currency, period, start_date,
  end_date.
- **RecurringExpense**: id, description, amount_minor, currency, frequency, next_occurrence,
  category_id, active.

## Natural-language entry

`finance.nlp.parse_transaction()` extracts amount+currency (via `Money.parse`), a date ("today"
default, "yesterday", "N days ago", "last \<weekday\>", or an explicit `YYYY-MM-DD`), a merchant
("at X"), and a description ("on X"/"for X") — all with plain regexes, no model call, so it's
identical every time it's run. `finance.service.FinanceService._add_transaction_from_draft()` then
resolves a category by matching the description/merchant against keyword rules
(`finance_category_rules`, seeded by `ensure_default_categories()` with a reasonable starting
taxonomy: Food, Transport, Entertainment, Shopping, Bills, Health, Education).

```
"I spent ₹340 on lunch"
  → amount=34000 (paise), currency=INR, occurred_at=<today>, description="Lunch"
  → category resolved via keyword "lunch" → "Food"
  → persisted, phrased back as "Logged 340.00 INR for Lunch (Food) on <date>."
```

## Natural-language queries

`finance.nlp.parse_query()` extracts a period ("this month" — the default when nothing is stated,
"last month", "this week", "this year", "today", "yesterday") and an optional category hint from
"on X"/"for X". `finance.service.FinanceService._run_query()` resolves the category (if any) and calls
`finance.analytics.category_total()` or `.period_summary()`, both of which sum `amount_minor` per
currency (never combining currencies) and return `Money` objects the service phrases into a sentence.

```
"How much did I spend on food this month?"
  → period = [first of this month, today], category_hint = "food" → "Food"
  → SUM(amount_minor) over matching finance_transactions, grouped by currency
  → "You spent 340.00 INR on Food between <start> and <end> (1 transaction)."
```

## Budgets, recurring expenses, import/export

- **Budgets**: `finance.analytics.budget_status()` sums actual spend in the budget's category/period
  and reports spent/remaining/percent_used/exceeded. `spending_alerts()` surfaces every budget that's
  exceeded or within a warn threshold (default 90%).
- **Recurring expenses**: `finance.recurring.next_occurrence()` computes the next date for daily/
  weekly/biweekly/monthly/yearly frequencies, correctly rolling month-end dates (Jan 31 → Feb 28, or
  29 in a leap year) instead of raising or silently wrapping. `FinanceService.apply_due_recurring()`
  materializes every due recurring expense as a real transaction and advances `next_occurrence`.
- **CSV import** (`finance.imports.csv_import`): caller-supplied column mapping (not tied to one
  bank's export format), per-row error collection (one bad row doesn't abort the batch), and dedup by
  a content hash of (amount, currency, date, merchant, description) so re-importing a statement is
  safe.
- **Export** (`finance.export`): CSV or JSON.
- **Receipt import** (`finance.imports.receipt`, Phase 2): reads a photographed/scanned receipt via
  `vision.document`, then parses and persists it through the exact same `Money.parse`/
  `normalize_date`/dedup path as every other entry method — see `docs/VISION.md` for the full pipeline
  and why letting vision *read* a printed amount doesn't compromise the "LLM never computes a total"
  rule above. **Statement import** (a PDF bank/card statement, as opposed to a CSV export already
  covered above) remains interface-only (`finance/imports/interfaces.py`) — it needs multi-page,
  multi-transaction document structure extraction beyond what single-receipt extraction does. Kanna
  never asks for bank login credentials for any of this.

## Deterministic test coverage

`tests/test_money.py`, `test_finance_nlp.py`, `test_finance_transactions.py`,
`test_finance_budgets.py`, `test_finance_recurring.py`, `test_finance_csv.py`,
`test_finance_dates.py`, `test_finance_receipt_import.py`, `test_finance_receipt_tool.py` cover
parsing/rounding, category assignment, date-boundary filtering, monthly/category totals,
multi-currency separation, budget status, month-end recurrence rollover, import dedup/error-
reporting, and receipt extraction (missing amount, missing/unparseable date, currency fallback,
duplicate detection, vision-provider failure) — all against `vision.document.fake.FakeDocumentProvider`,
never the network. See `docs/TESTING.md`.
