"""SQLite-backed repository for the finance subsystem.

All reads and writes to `finance_*` tables go through here. Nothing in
this file computes a derived total that's more than a straightforward
`SUM`/`COUNT` over `amount_minor` — analytics that combine multiple
queries or apply business rules live in `finance/analytics.py`, and even
those never touch anything but SQL aggregation and `Decimal`.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from core.memory.db import Database
from finance.models import Budget, Category, RecurringExpense, Transaction


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_hash(*, amount_minor: int, currency: str, occurred_at: str, merchant: str | None,
                  description: str | None) -> str:
    """A stable hash used to dedupe imported transactions (CSV/statement import)."""
    raw = f"{amount_minor}|{currency}|{occurred_at}|{merchant or ''}|{description or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CategoryRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, name: str, parent_category: str | None = None) -> Category:
        category_id = str(uuid.uuid4())
        created_at = _now()
        self.db.execute(
            "INSERT INTO finance_categories (id, name, parent_category, created_at) VALUES (?, ?, ?, ?)",
            (category_id, name, parent_category, created_at),
        )
        return Category(id=category_id, name=name, parent_category=parent_category, created_at=created_at)

    def get_or_create(self, name: str, parent_category: str | None = None) -> Category:
        existing = self.get_by_name(name)
        if existing is not None:
            return existing
        return self.create(name, parent_category)

    def get(self, category_id: str) -> Category | None:
        row = self.db.query_one("SELECT * FROM finance_categories WHERE id = ?", (category_id,))
        return None if row is None else self._from_row(row)

    def get_by_name(self, name: str) -> Category | None:
        row = self.db.query_one(
            "SELECT * FROM finance_categories WHERE name = ? COLLATE NOCASE", (name,)
        )
        return None if row is None else self._from_row(row)

    def list(self) -> list[Category]:
        rows = self.db.query("SELECT * FROM finance_categories ORDER BY name ASC")
        return [self._from_row(r) for r in rows]

    def add_keyword_rule(self, keyword: str, category_id: str) -> None:
        self.db.execute(
            "INSERT INTO finance_category_rules (keyword, category_id) VALUES (?, ?)",
            (keyword.lower(), category_id),
        )

    def find_category_for_keyword(self, text: str) -> Category | None:
        """Return the category whose keyword rule matches the longest substring of `text`."""
        text_lower = text.lower()
        rows = self.db.query(
            "SELECT r.keyword, c.* FROM finance_category_rules r "
            "JOIN finance_categories c ON c.id = r.category_id"
        )
        best: tuple[int, Category] | None = None
        for row in rows:
            keyword = row["keyword"]
            if keyword in text_lower:
                if best is None or len(keyword) > best[0]:
                    best = (len(keyword), self._from_row(row))
        return best[1] if best else None

    @staticmethod
    def _from_row(row) -> Category:
        return Category(id=row["id"], name=row["name"], parent_category=row["parent_category"],
                         created_at=row["created_at"])


class TransactionRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, tx: Transaction) -> Transaction:
        tx_id = tx.id or str(uuid.uuid4())
        now = _now()
        chash = tx.content_hash or content_hash(
            amount_minor=tx.amount_minor, currency=tx.currency, occurred_at=tx.occurred_at,
            merchant=tx.merchant, description=tx.description,
        )
        self.db.execute(
            "INSERT INTO finance_transactions "
            "(id, amount_minor, currency, occurred_at, category_id, subcategory, merchant, "
            " description, payment_method, source, notes, content_hash, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (tx_id, tx.amount_minor, tx.currency, tx.occurred_at, tx.category_id, tx.subcategory,
             tx.merchant, tx.description, tx.payment_method, tx.source, tx.notes, chash, now, now),
        )
        tx.id, tx.content_hash, tx.created_at, tx.updated_at = tx_id, chash, now, now
        return tx

    def exists_with_hash(self, chash: str) -> bool:
        row = self.db.query_one("SELECT 1 FROM finance_transactions WHERE content_hash = ?", (chash,))
        return row is not None

    def get(self, tx_id: str) -> Transaction | None:
        row = self.db.query_one("SELECT * FROM finance_transactions WHERE id = ?", (tx_id,))
        return None if row is None else self._from_row(row)

    def delete(self, tx_id: str) -> bool:
        cur = self.db.execute("DELETE FROM finance_transactions WHERE id = ?", (tx_id,))
        return cur.rowcount > 0

    def search(self, *, start_date: str | None = None, end_date: str | None = None,
               category_id: str | None = None, currency: str | None = None,
               merchant_contains: str | None = None, text_contains: str | None = None,
               limit: int = 500) -> list[Transaction]:
        clauses: list[str] = []
        params: list = []
        if start_date:
            clauses.append("occurred_at >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("occurred_at <= ?")
            params.append(end_date)
        if category_id:
            clauses.append("category_id = ?")
            params.append(category_id)
        if currency:
            clauses.append("currency = ?")
            params.append(currency.upper())
        if merchant_contains:
            clauses.append("merchant LIKE ?")
            params.append(f"%{merchant_contains}%")
        if text_contains:
            clauses.append("(description LIKE ? OR merchant LIKE ? OR notes LIKE ?)")
            like = f"%{text_contains}%"
            params.extend([like, like, like])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM finance_transactions {where} ORDER BY occurred_at DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = self.db.query(sql, tuple(params))
        return [self._from_row(r) for r in rows]

    @staticmethod
    def _from_row(row) -> Transaction:
        return Transaction(
            id=row["id"], amount_minor=row["amount_minor"], currency=row["currency"],
            occurred_at=row["occurred_at"], category_id=row["category_id"],
            subcategory=row["subcategory"], merchant=row["merchant"], description=row["description"],
            payment_method=row["payment_method"], source=row["source"], notes=row["notes"],
            content_hash=row["content_hash"], created_at=row["created_at"], updated_at=row["updated_at"],
        )


class BudgetRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, budget: Budget) -> Budget:
        budget_id = budget.id or str(uuid.uuid4())
        now = _now()
        self.db.execute(
            "INSERT INTO finance_budgets "
            "(id, category_id, amount_minor, currency, period, start_date, end_date, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (budget_id, budget.category_id, budget.amount_minor, budget.currency, budget.period,
             budget.start_date, budget.end_date, now, now),
        )
        budget.id, budget.created_at, budget.updated_at = budget_id, now, now
        return budget

    def list(self, *, category_id: str | None = None) -> list[Budget]:
        if category_id:
            rows = self.db.query("SELECT * FROM finance_budgets WHERE category_id = ?", (category_id,))
        else:
            rows = self.db.query("SELECT * FROM finance_budgets ORDER BY start_date DESC")
        return [self._from_row(r) for r in rows]

    def get(self, budget_id: str) -> Budget | None:
        row = self.db.query_one("SELECT * FROM finance_budgets WHERE id = ?", (budget_id,))
        return None if row is None else self._from_row(row)

    @staticmethod
    def _from_row(row) -> Budget:
        return Budget(id=row["id"], category_id=row["category_id"], amount_minor=row["amount_minor"],
                       currency=row["currency"], period=row["period"], start_date=row["start_date"],
                       end_date=row["end_date"], created_at=row["created_at"], updated_at=row["updated_at"])


class RecurringRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, rec: RecurringExpense) -> RecurringExpense:
        rec_id = rec.id or str(uuid.uuid4())
        now = _now()
        self.db.execute(
            "INSERT INTO finance_recurring "
            "(id, description, amount_minor, currency, frequency, next_occurrence, category_id, "
            " active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (rec_id, rec.description, rec.amount_minor, rec.currency, rec.frequency,
             rec.next_occurrence, rec.category_id, 1 if rec.active else 0, now, now),
        )
        rec.id, rec.created_at, rec.updated_at = rec_id, now, now
        return rec

    def list(self, *, active_only: bool = False) -> list[RecurringExpense]:
        if active_only:
            rows = self.db.query("SELECT * FROM finance_recurring WHERE active = 1 ORDER BY next_occurrence ASC")
        else:
            rows = self.db.query("SELECT * FROM finance_recurring ORDER BY next_occurrence ASC")
        return [self._from_row(r) for r in rows]

    def due(self, as_of: str) -> list[RecurringExpense]:
        rows = self.db.query(
            "SELECT * FROM finance_recurring WHERE active = 1 AND next_occurrence <= ? "
            "ORDER BY next_occurrence ASC",
            (as_of,),
        )
        return [self._from_row(r) for r in rows]

    def update_next_occurrence(self, rec_id: str, next_occurrence: str) -> None:
        self.db.execute(
            "UPDATE finance_recurring SET next_occurrence = ?, updated_at = ? WHERE id = ?",
            (next_occurrence, _now(), rec_id),
        )

    def set_active(self, rec_id: str, active: bool) -> None:
        self.db.execute(
            "UPDATE finance_recurring SET active = ?, updated_at = ? WHERE id = ?",
            (1 if active else 0, _now(), rec_id),
        )

    @staticmethod
    def _from_row(row) -> RecurringExpense:
        return RecurringExpense(
            id=row["id"], description=row["description"], amount_minor=row["amount_minor"],
            currency=row["currency"], frequency=row["frequency"], next_occurrence=row["next_occurrence"],
            category_id=row["category_id"], active=bool(row["active"]),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )
