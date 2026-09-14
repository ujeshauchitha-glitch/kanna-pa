"""Deterministic finance calculations.

Every function here does its arithmetic with `Money`/integers over rows
read from the repository — nothing is estimated, inferred, or produced by
an LLM. This is the only place allowed to answer "how much did I spend".
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from finance.models import Budget, Transaction
from finance.money import Money
from finance.repository import BudgetRepository, CategoryRepository, TransactionRepository


def _group_by_currency(transactions: list[Transaction]) -> dict[str, Money]:
    totals: dict[str, int] = defaultdict(int)
    for tx in transactions:
        totals[tx.currency] += tx.amount_minor
    return {currency: Money(amount, currency) for currency, amount in totals.items()}


@dataclass
class PeriodSummary:
    start_date: str
    end_date: str
    totals_by_currency: dict[str, Money]
    by_category: dict[str, dict[str, Money]]  # category name -> currency -> Money
    transaction_count: int


def period_summary(tx_repo: TransactionRepository, cat_repo: CategoryRepository, *,
                    start_date: str, end_date: str, currency: str | None = None) -> PeriodSummary:
    transactions = tx_repo.search(start_date=start_date, end_date=end_date, currency=currency,
                                   limit=100_000)
    totals = _group_by_currency(transactions)

    by_category_tx: dict[str, list[Transaction]] = defaultdict(list)
    category_names: dict[str | None, str] = {None: "Uncategorized"}
    for cat in cat_repo.list():
        category_names[cat.id] = cat.name
    for tx in transactions:
        name = category_names.get(tx.category_id, "Uncategorized")
        by_category_tx[name].append(tx)

    by_category = {name: _group_by_currency(txs) for name, txs in by_category_tx.items()}

    return PeriodSummary(
        start_date=start_date, end_date=end_date, totals_by_currency=totals,
        by_category=by_category, transaction_count=len(transactions),
    )


def category_total(tx_repo: TransactionRepository, *, category_id: str, start_date: str,
                    end_date: str, currency: str | None = None) -> dict[str, Money]:
    transactions = tx_repo.search(start_date=start_date, end_date=end_date, category_id=category_id,
                                   currency=currency, limit=100_000)
    return _group_by_currency(transactions)


@dataclass
class BudgetStatus:
    budget: Budget
    spent: Money
    remaining: Money
    percent_used: float
    exceeded: bool


def budget_status(tx_repo: TransactionRepository, budget: Budget, *, as_of_date: str) -> BudgetStatus:
    end = budget.end_date or as_of_date
    transactions = tx_repo.search(
        start_date=budget.start_date, end_date=min(end, as_of_date), category_id=budget.category_id,
        currency=budget.currency, limit=100_000,
    )
    spent_minor = sum(tx.amount_minor for tx in transactions)
    spent = Money(spent_minor, budget.currency)
    budget_amount = Money(budget.amount_minor, budget.currency)
    remaining = budget_amount - spent
    percent = (spent_minor / budget.amount_minor * 100) if budget.amount_minor else 0.0
    return BudgetStatus(budget=budget, spent=spent, remaining=remaining, percent_used=percent,
                         exceeded=spent_minor > budget.amount_minor)


def spending_alerts(budget_repo: BudgetRepository, tx_repo: TransactionRepository, *,
                     as_of_date: str, warn_threshold_percent: float = 90.0) -> list[BudgetStatus]:
    """Budgets that are exceeded or within `warn_threshold_percent` of their limit."""
    alerts = []
    for budget in budget_repo.list():
        status = budget_status(tx_repo, budget, as_of_date=as_of_date)
        if status.exceeded or status.percent_used >= warn_threshold_percent:
            alerts.append(status)
    return alerts
