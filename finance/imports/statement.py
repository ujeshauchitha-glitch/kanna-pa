"""Bank/card statement import: the real implementation of `StatementImporter`.

Same discipline as `receipt.py`: `vision.document` only reads what's
printed; every parse (`Money.parse`, `normalize_date`) and every
database write happens here.

**Scope note:** Kanna's finance subsystem tracks *spending* — every
other entry path (NL, CSV, receipt) records a purchase. A statement
mixes debits (spend — fits the model) and credits (deposits, refunds,
incoming transfers — money *in*, which the current `Transaction` model
has no representation for; there is no signed-amount or income/expense
distinction). Rather than force a credit in with a fabricated sign, or
silently drop it, credit rows (and rows whose direction the model
couldn't determine) are reported in `ImportResult.errors` as an
explained skip — visible, never silently lost, never misrepresented as
a purchase. Only debit rows become transactions. If Kanna grows income
tracking later, this is the boundary that decision would move.
"""
from __future__ import annotations

from datetime import date

from core.errors import InvalidMoneyAmount
from finance.dates import normalize_date
from finance.imports.csv_import import ImportError_, ImportResult
from finance.models import Transaction
from finance.money import Money
from finance.repository import CategoryRepository, TransactionRepository, content_hash
from vision.document.base import DocumentProvider


def import_statement(file_bytes: bytes, *, mime_type: str, tx_repo: TransactionRepository,
                      cat_repo: CategoryRepository, provider: DocumentProvider,
                      default_currency: str, source: str = "statement_ocr",
                      today: str | None = None) -> ImportResult:
    result = ImportResult()
    today = today or date.today().isoformat()

    try:
        extraction = provider.extract_statement(file_bytes, mime_type=mime_type)
    except Exception as exc:  # noqa: BLE001 - a provider failure (auth, network) is one reportable row
        result.errors.append(ImportError_(row_number=0, error=f"vision provider failed: {exc}", raw_row={}))
        return result

    if not extraction.transactions:
        detail = f" ({extraction.notes})" if extraction.notes else ""
        result.errors.append(ImportError_(
            row_number=0, error=f"no transactions found on the statement{detail}",
            raw_row={"raw_text": extraction.raw_text[:500]},
        ))
        return result

    currency = extraction.account_currency or default_currency

    for row_number, row in enumerate(extraction.transactions, start=1):
        raw_row = {"description": row.description or "", "amount": row.amount_text or "",
                   "date": row.date_text or "", "direction": row.direction or ""}

        if row.direction == "credit":
            result.errors.append(ImportError_(
                row_number=row_number,
                error="credit transaction — Kanna currently tracks spending only; not imported",
                raw_row=raw_row,
            ))
            continue
        if row.direction != "debit":
            result.errors.append(ImportError_(
                row_number=row_number,
                error="transaction direction unclear — not imported, to avoid misrepresenting a "
                      "credit as spend",
                raw_row=raw_row,
            ))
            continue
        if not row.amount_text:
            result.errors.append(ImportError_(row_number=row_number, error="missing amount", raw_row=raw_row))
            continue

        try:
            money = Money.parse(row.amount_text, currency)
        except InvalidMoneyAmount as exc:
            result.errors.append(ImportError_(
                row_number=row_number, error=f"could not parse amount {row.amount_text!r}: {exc}",
                raw_row=raw_row,
            ))
            continue

        occurred_at = today
        notes = None
        if row.date_text:
            try:
                occurred_at = normalize_date(row.date_text)
            except ValueError:
                notes = f"could not parse statement date {row.date_text!r}; used today's date"
        else:
            notes = "no date found for this row; used today's date"

        category_id = None
        if row.description:
            category = cat_repo.find_category_for_keyword(row.description)
            if category is not None:
                category_id = category.id

        chash = content_hash(amount_minor=money.amount_minor, currency=money.currency,
                              occurred_at=occurred_at, merchant=None, description=row.description)
        if tx_repo.exists_with_hash(chash):
            result.skipped_duplicates += 1
            continue

        tx = tx_repo.create(Transaction(
            id="", amount_minor=money.amount_minor, currency=money.currency,
            occurred_at=occurred_at, category_id=category_id, description=row.description,
            source=source, notes=notes,
        ))
        result.created += 1
        result.created_transaction_ids.append(tx.id)

    return result
