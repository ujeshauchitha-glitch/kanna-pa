"""Finance domain models. Plain dataclasses — the repository maps these to/from SQLite rows."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Category:
    id: str
    name: str
    parent_category: str | None = None
    created_at: str = ""


@dataclass
class Transaction:
    id: str
    amount_minor: int
    currency: str
    occurred_at: str  # YYYY-MM-DD (local calendar date)
    category_id: str | None = None
    subcategory: str | None = None
    merchant: str | None = None
    description: str | None = None
    payment_method: str | None = None
    source: str = "manual"
    notes: str | None = None
    content_hash: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Budget:
    id: str
    category_id: str | None
    amount_minor: int
    currency: str
    period: str  # "monthly" | "weekly" | "yearly" | "custom"
    start_date: str
    end_date: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class RecurringExpense:
    id: str
    description: str
    amount_minor: int
    currency: str
    frequency: str  # "daily" | "weekly" | "biweekly" | "monthly" | "yearly"
    next_occurrence: str
    category_id: str | None = None
    active: bool = True
    created_at: str = ""
    updated_at: str = ""
