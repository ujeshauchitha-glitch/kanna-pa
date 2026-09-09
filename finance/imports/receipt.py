"""Receipt import: the real implementation the Phase 1 `ReceiptImporter` interface was waiting on.

Same discipline as `csv_import.py`: a `vision.document.base.DocumentProvider`
reads the receipt and returns *raw, unparsed* text fields (see
`ReceiptExtraction`'s docstring for why) — this module does every bit of
the actual parsing (`Money.parse`, `normalize_date`) and every database
write. The vision provider is extraction only; it never computes
anything that ends up in `finance.analytics`'s aggregation path.

Follows `import_csv`'s functional style (explicit repositories passed
in, not a class) and reuses its `ImportResult`/`ImportError_` shape so
callers (the `finance_import_receipt` tool, in particular) handle both
the same way.
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


def import_receipt(file_bytes: bytes, *, mime_type: str, tx_repo: TransactionRepository,
                    cat_repo: CategoryRepository, provider: DocumentProvider,
                    default_currency: str, source: str = "receipt_ocr",
                    today: str | None = None) -> ImportResult:
    result = ImportResult()
    today = today or date.today().isoformat()

    try:
        extraction = provider.extract_receipt(file_bytes, mime_type=mime_type)
    except Exception as exc:  # noqa: BLE001 - a provider failure (auth, network) is one reportable row
        result.errors.append(ImportError_(row_number=1, error=f"vision provider failed: {exc}", raw_row={}))
        return result

    if not extraction.total_amount_text:
        detail = f" ({extraction.notes})" if extraction.notes else ""
        result.errors.append(ImportError_(
            row_number=1, error=f"could not find a total amount on the receipt{detail}",
            raw_row={"raw_text": extraction.raw_text[:500]},
        ))
        return result

    currency = extraction.currency_code or default_currency
    try:
        money = Money.parse(extraction.total_amount_text, currency)
    except InvalidMoneyAmount as exc:
        result.errors.append(ImportError_(
            row_number=1, error=f"could not parse total amount {extraction.total_amount_text!r}: {exc}",
            raw_row={"raw_text": extraction.raw_text[:500]},
        ))
        return result

    notes_parts: list[str] = []
    occurred_at = today
    if extraction.date_text:
        try:
            occurred_at = normalize_date(extraction.date_text)
        except ValueError:
            notes_parts.append(f"could not parse receipt date {extraction.date_text!r}; used today's date")
    else:
        notes_parts.append("no date found on receipt; used today's date")

    if extraction.notes:
        notes_parts.append(extraction.notes)

    category_id = None
    if extraction.merchant:
        category = cat_repo.find_category_for_keyword(extraction.merchant)
        if category is not None:
            category_id = category.id

    chash = content_hash(amount_minor=money.amount_minor, currency=money.currency,
                          occurred_at=occurred_at, merchant=extraction.merchant, description=None)
    if tx_repo.exists_with_hash(chash):
        result.skipped_duplicates += 1
        return result

    tx = tx_repo.create(Transaction(
        id="", amount_minor=money.amount_minor, currency=money.currency, occurred_at=occurred_at,
        category_id=category_id, merchant=extraction.merchant, source=source,
        notes="; ".join(notes_parts) or None,
    ))
    result.created = 1
    result.created_transaction_ids.append(tx.id)
    return result
