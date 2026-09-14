from __future__ import annotations

from finance import analytics
from finance.service import FinanceService


def test_budget_status_under_budget(db):
    svc = FinanceService(db, default_currency="INR")
    svc.set_budget(category_name="Food", amount_minor=100_00, currency="INR",
                    period="monthly", start_date="2024-03-01")
    svc.add_transaction(amount_minor=30_00, currency="INR", occurred_at="2024-03-05",
                         category_name="Food")

    budget = svc.budgets.list()[0]
    status = analytics.budget_status(svc.transactions, budget, as_of_date="2024-03-31")
    assert status.spent.amount_minor == 3000
    assert status.remaining.amount_minor == 7000
    assert not status.exceeded


def test_budget_status_exceeded(db):
    svc = FinanceService(db, default_currency="INR")
    svc.set_budget(category_name="Food", amount_minor=100_00, currency="INR",
                    period="monthly", start_date="2024-03-01")
    svc.add_transaction(amount_minor=60_00, currency="INR", occurred_at="2024-03-05",
                         category_name="Food")
    svc.add_transaction(amount_minor=60_00, currency="INR", occurred_at="2024-03-10",
                         category_name="Food")

    budget = svc.budgets.list()[0]
    status = analytics.budget_status(svc.transactions, budget, as_of_date="2024-03-31")
    assert status.exceeded
    assert status.percent_used > 100


def test_spending_alerts_surface_exceeded_budgets(db):
    svc = FinanceService(db, default_currency="INR")
    svc.set_budget(category_name="Food", amount_minor=100_00, currency="INR",
                    period="monthly", start_date="2024-03-01")
    svc.set_budget(category_name="Transport", amount_minor=100_00, currency="INR",
                    period="monthly", start_date="2024-03-01")
    svc.add_transaction(amount_minor=150_00, currency="INR", occurred_at="2024-03-05",
                         category_name="Food")
    svc.add_transaction(amount_minor=10_00, currency="INR", occurred_at="2024-03-05",
                         category_name="Transport")

    alerts = analytics.spending_alerts(svc.budgets, svc.transactions, as_of_date="2024-03-31")
    assert len(alerts) == 1
    assert alerts[0].budget.category_id == svc.categories.get_by_name("Food").id


def test_budget_transactions_outside_category_are_excluded(db):
    svc = FinanceService(db, default_currency="INR")
    svc.set_budget(category_name="Food", amount_minor=100_00, currency="INR",
                    period="monthly", start_date="2024-03-01")
    svc.add_transaction(amount_minor=50_00, currency="INR", occurred_at="2024-03-05",
                         category_name="Transport")

    budget = svc.budgets.list()[0]
    status = analytics.budget_status(svc.transactions, budget, as_of_date="2024-03-31")
    assert status.spent.amount_minor == 0
