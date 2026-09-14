"""Finance orchestration: NL -> structured intent -> DB -> deterministic result -> phrasing.

This is the only layer allowed to turn a sentence into a database write
or a spoken-language answer. The path is always:

    natural language
      -> finance.nlp (deterministic parse; an LLMPlanner may substitute
         its own structured draft/query here, but never a raw number)
      -> TransactionDraft / FinanceQuery
      -> finance.repository (SQL)
      -> finance.analytics (SQL aggregation / Decimal arithmetic)
      -> a plain-language sentence built from that already-computed number

The LLM is never in the arithmetic path.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.memory.db import Database
from finance import analytics, nlp
from finance.models import Budget, RecurringExpense, Transaction
from finance.money import Money
from finance.repository import BudgetRepository, CategoryRepository, RecurringRepository, TransactionRepository

# name -> keywords used to seed category inference. Extend freely; this is
# just a reasonable starting point, not a fixed taxonomy.
DEFAULT_CATEGORIES: dict[str, list[str]] = {
    "Food": ["lunch", "dinner", "breakfast", "coffee", "restaurant", "cafe", "grocery",
              "groceries", "snack", "food", "pizza", "takeout"],
    "Transport": ["uber", "taxi", "cab", "bus", "train", "fuel", "petrol", "diesel",
                   "metro", "parking", "auto"],
    "Entertainment": ["movie", "cinema", "netflix", "spotify", "concert", "game", "gaming"],
    "Shopping": ["clothes", "clothing", "shoes", "amazon", "shopping", "electronics"],
    "Bills": ["rent", "electricity", "water bill", "internet", "wifi", "phone bill", "subscription"],
    "Health": ["pharmacy", "medicine", "doctor", "hospital", "clinic", "gym"],
    "Education": ["tuition", "books", "course", "stationery"],
}


def ensure_default_categories(db: Database) -> None:
    """Seed the default category set + keyword rules if the categories table is empty. Idempotent."""
    cat_repo = CategoryRepository(db)
    if cat_repo.list():
        return
    for name, keywords in DEFAULT_CATEGORIES.items():
        category = cat_repo.create(name)
        for keyword in keywords:
            cat_repo.add_keyword_rule(keyword, category.id)


@dataclass
class AddTransactionResult:
    transaction: Transaction
    category_name: str | None
    message: str


@dataclass
class QueryResult:
    query: nlp.FinanceQuery
    summary: analytics.PeriodSummary
    message: str


class FinanceService:
    def __init__(self, db: Database, *, default_currency: str = "INR", timezone_name: str = "UTC") -> None:
        self.db = db
        self.default_currency = default_currency
        self.timezone_name = timezone_name
        self.categories = CategoryRepository(db)
        self.transactions = TransactionRepository(db)
        self.budgets = BudgetRepository(db)
        self.recurring = RecurringRepository(db)
        ensure_default_categories(db)

    # -- Transaction entry ------------------------------------------------

    def add_transaction_from_text(self, text: str) -> AddTransactionResult:
        draft = nlp.parse_transaction(text, default_currency=self.default_currency,
                                       timezone_name=self.timezone_name)
        return self._add_transaction_from_draft(draft, source="nlp")

    def add_transaction(self, *, amount_minor: int, currency: str, occurred_at: str,
                         description: str | None = None, merchant: str | None = None,
                         category_name: str | None = None, notes: str | None = None,
                         source: str = "manual") -> AddTransactionResult:
        draft = nlp.TransactionDraft(amount_minor=amount_minor, currency=currency,
                                      occurred_at=occurred_at, description=description,
                                      merchant=merchant, category_hint=category_name)
        return self._add_transaction_from_draft(draft, source=source, explicit_category=category_name,
                                                  notes=notes)

    def _add_transaction_from_draft(self, draft: nlp.TransactionDraft, *, source: str,
                                      explicit_category: str | None = None,
                                      notes: str | None = None) -> AddTransactionResult:
        category = None
        if explicit_category:
            category = self.categories.get_or_create(explicit_category)
        elif draft.category_hint:
            category = self.categories.find_category_for_keyword(draft.category_hint)

        tx = self.transactions.create(Transaction(
            id="", amount_minor=draft.amount_minor, currency=draft.currency,
            occurred_at=draft.occurred_at, category_id=category.id if category else None,
            merchant=draft.merchant, description=draft.description, source=source, notes=notes,
        ))

        money = Money(tx.amount_minor, tx.currency)
        cat_part = f" ({category.name})" if category else ""
        desc_part = f" for {tx.description}" if tx.description else ""
        message = f"Logged {money}{desc_part}{cat_part} on {tx.occurred_at}."
        return AddTransactionResult(transaction=tx, category_name=category.name if category else None,
                                     message=message)

    # -- Queries ------------------------------------------------------------

    def query_from_text(self, text: str) -> QueryResult:
        query = nlp.parse_query(text, timezone_name=self.timezone_name)
        return self._run_query(query)

    def _run_query(self, query: nlp.FinanceQuery) -> QueryResult:
        category_id = None
        category_name = None
        if query.category_hint:
            category = self.categories.find_category_for_keyword(query.category_hint)
            if category is not None:
                category_id = category.id
                category_name = category.name

        if category_id:
            totals = analytics.category_total(self.transactions, category_id=category_id,
                                                start_date=query.start_date, end_date=query.end_date,
                                                currency=query.currency)
            summary = analytics.PeriodSummary(
                start_date=query.start_date, end_date=query.end_date, totals_by_currency=totals,
                by_category={category_name or "Uncategorized": totals},
                transaction_count=sum(1 for _ in self.transactions.search(
                    start_date=query.start_date, end_date=query.end_date, category_id=category_id,
                    currency=query.currency, limit=100_000,
                )),
            )
        else:
            summary = analytics.period_summary(self.transactions, self.categories,
                                                 start_date=query.start_date, end_date=query.end_date,
                                                 currency=query.currency)

        message = self._phrase_summary(summary, category_name)
        return QueryResult(query=query, summary=summary, message=message)

    @staticmethod
    def _phrase_summary(summary: analytics.PeriodSummary, category_name: str | None) -> str:
        if not summary.totals_by_currency:
            scope = f" on {category_name}" if category_name else ""
            return f"No spending{scope} found between {summary.start_date} and {summary.end_date}."

        parts = [str(money) for money in summary.totals_by_currency.values()]
        scope = f" on {category_name}" if category_name else ""
        return (f"You spent {', '.join(parts)}{scope} between {summary.start_date} and "
                f"{summary.end_date} ({summary.transaction_count} transaction"
                f"{'s' if summary.transaction_count != 1 else ''}).")

    # -- Budgets --------------------------------------------------------

    def set_budget(self, *, category_name: str | None, amount_minor: int, currency: str,
                    period: str, start_date: str, end_date: str | None = None) -> Budget:
        category_id = self.categories.get_or_create(category_name).id if category_name else None
        return self.budgets.create(Budget(
            id="", category_id=category_id, amount_minor=amount_minor, currency=currency,
            period=period, start_date=start_date, end_date=end_date,
        ))

    def budget_report(self, *, as_of_date: str) -> list[analytics.BudgetStatus]:
        return [analytics.budget_status(self.transactions, b, as_of_date=as_of_date)
                for b in self.budgets.list()]

    # -- Recurring --------------------------------------------------------

    def add_recurring(self, *, description: str, amount_minor: int, currency: str, frequency: str,
                       next_occurrence: str, category_name: str | None = None) -> RecurringExpense:
        category_id = self.categories.get_or_create(category_name).id if category_name else None
        return self.recurring.create(RecurringExpense(
            id="", description=description, amount_minor=amount_minor, currency=currency,
            frequency=frequency, next_occurrence=next_occurrence, category_id=category_id,
        ))

    def apply_due_recurring(self, *, as_of_date: str) -> list[Transaction]:
        """Materialize every due recurring expense as a transaction and advance its next_occurrence."""
        from datetime import date as _date

        from finance.recurring import next_occurrence as compute_next

        created = []
        for rec in self.recurring.due(as_of_date):
            tx = self.transactions.create(Transaction(
                id="", amount_minor=rec.amount_minor, currency=rec.currency,
                occurred_at=rec.next_occurrence, category_id=rec.category_id,
                description=rec.description, source="recurring",
            ))
            created.append(tx)
            new_next = compute_next(_date.fromisoformat(rec.next_occurrence), rec.frequency)
            self.recurring.update_next_occurrence(rec.id, new_next.isoformat())
        return created

    # -- Search / export --------------------------------------------------

    def search(self, **kwargs) -> list[Transaction]:
        return self.transactions.search(**kwargs)
