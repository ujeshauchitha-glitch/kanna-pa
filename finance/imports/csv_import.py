"""CSV transaction import.

Reads rows via a caller-supplied column mapping (so it isn't tied to one
bank's export format), parses each into a `Transaction`, and dedupes
against existing rows by content hash so re-importing the same statement
twice doesn't double-count.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from finance.dates import normalize_date
from finance.models import Transaction
from finance.money import Money
from finance.repository import CategoryRepository, TransactionRepository, content_hash

DEFAULT_COLUMN_MAP = {
    "date": "date",
    "amount": "amount",
    "description": "description",
    "merchant": "merchant",
    "category": "category",
}


@dataclass
class ImportError_:
    row_number: int
    error: str
    raw_row: dict


@dataclass
class ImportResult:
    created: int = 0
    skipped_duplicates: int = 0
    errors: list[ImportError_] = field(default_factory=list)
    created_transaction_ids: list[str] = field(default_factory=list)


def import_csv(csv_text: str, *, tx_repo: TransactionRepository, cat_repo: CategoryRepository,
                default_currency: str, column_map: dict[str, str] | None = None,
                source: str = "csv_import") -> ImportResult:
    mapping = {**DEFAULT_COLUMN_MAP, **(column_map or {})}
    result = ImportResult()

    reader = csv.DictReader(io.StringIO(csv_text))
    for row_number, row in enumerate(reader, start=2):  # header is row 1
        try:
            date_str = (row.get(mapping["date"]) or "").strip()
            amount_str = (row.get(mapping["amount"]) or "").strip()
            description = (row.get(mapping.get("description", "")) or "").strip() or None
            merchant = (row.get(mapping.get("merchant", "")) or "").strip() or None
            category_name = (row.get(mapping.get("category", "")) or "").strip() or None

            if not date_str:
                raise ValueError("missing date")
            if not amount_str:
                raise ValueError("missing amount")

            occurred_at = normalize_date(date_str)

            try:
                money = Money.from_decimal(Decimal(amount_str.replace(",", "")), default_currency)
            except InvalidOperation as exc:
                raise ValueError(f"unparseable amount: {amount_str!r}") from exc

            category_id = None
            if category_name:
                category_id = cat_repo.get_or_create(category_name).id

            chash = content_hash(amount_minor=money.amount_minor, currency=money.currency,
                                  occurred_at=occurred_at, merchant=merchant, description=description)
            if tx_repo.exists_with_hash(chash):
                result.skipped_duplicates += 1
                continue

            tx = tx_repo.create(Transaction(
                id="", amount_minor=money.amount_minor, currency=money.currency,
                occurred_at=occurred_at, category_id=category_id, merchant=merchant,
                description=description, source=source,
            ))
            result.created += 1
            result.created_transaction_ids.append(tx.id)
        except Exception as exc:  # noqa: BLE001 - one bad row must not abort the whole import
            result.errors.append(ImportError_(row_number=row_number, error=str(exc), raw_row=dict(row)))

    return result
