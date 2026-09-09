"""Export transactions to CSV or JSON."""
from __future__ import annotations

import csv
import io
import json

from finance.models import Transaction


def to_csv(transactions: list[Transaction]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "date", "amount_minor", "currency", "category_id", "subcategory", "merchant",
        "description", "payment_method", "source", "notes",
    ])
    for tx in transactions:
        writer.writerow([
            tx.id, tx.occurred_at, tx.amount_minor, tx.currency, tx.category_id or "",
            tx.subcategory or "", tx.merchant or "", tx.description or "", tx.payment_method or "",
            tx.source, tx.notes or "",
        ])
    return buf.getvalue()


def to_json(transactions: list[Transaction]) -> str:
    return json.dumps([
        {
            "id": tx.id, "date": tx.occurred_at, "amount_minor": tx.amount_minor,
            "currency": tx.currency, "category_id": tx.category_id, "subcategory": tx.subcategory,
            "merchant": tx.merchant, "description": tx.description,
            "payment_method": tx.payment_method, "source": tx.source, "notes": tx.notes,
        }
        for tx in transactions
    ], indent=2)
