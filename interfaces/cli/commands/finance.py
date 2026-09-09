"""`kanna finance ...` subcommands."""
from __future__ import annotations

import argparse
import sys

from core.bootstrap import Kanna
from finance.imports.csv_import import import_csv
from finance.service import FinanceService


def register(subparsers: argparse._SubParsersAction) -> None:
    finance_parser = subparsers.add_parser("finance", help="Personal finance tracking")
    finance_sub = finance_parser.add_subparsers(dest="finance_command", required=True)

    add_p = finance_sub.add_parser("add", help="Log a transaction from natural language")
    add_p.add_argument("text", help="e.g. 'I spent ₹340 on lunch'")
    add_p.set_defaults(func=_cmd_add)

    query_p = finance_sub.add_parser("query", help="Ask a spending question")
    query_p.add_argument("text", help="e.g. 'How much did I spend on food this month?'")
    query_p.set_defaults(func=_cmd_query)

    budget_p = finance_sub.add_parser("budget", help="Set a budget")
    budget_p.add_argument("--category", default="")
    budget_p.add_argument("--amount", required=True, type=float, help="Amount in major units, e.g. 5000")
    budget_p.add_argument("--currency", default=None)
    budget_p.add_argument("--period", default="monthly", choices=["monthly", "weekly", "yearly", "custom"])
    budget_p.add_argument("--start-date", required=True)
    budget_p.add_argument("--end-date", default=None)
    budget_p.set_defaults(func=_cmd_budget)

    recurring_p = finance_sub.add_parser("recurring", help="Register a recurring expense")
    recurring_p.add_argument("description")
    recurring_p.add_argument("--amount", required=True, type=float)
    recurring_p.add_argument("--currency", default=None)
    recurring_p.add_argument("--frequency", required=True,
                              choices=["daily", "weekly", "biweekly", "monthly", "yearly"])
    recurring_p.add_argument("--next-occurrence", required=True)
    recurring_p.add_argument("--category", default="")
    recurring_p.set_defaults(func=_cmd_recurring)

    import_p = finance_sub.add_parser("import", help="Import transactions from a CSV file")
    import_p.add_argument("path")
    import_p.set_defaults(func=_cmd_import)

    export_p = finance_sub.add_parser("export", help="Export transactions")
    export_p.add_argument("--start-date", default=None)
    export_p.add_argument("--end-date", default=None)
    export_p.add_argument("--format", default="csv", choices=["csv", "json"])
    export_p.set_defaults(func=_cmd_export)


def _service(kanna: Kanna) -> FinanceService:
    return FinanceService(kanna.db, default_currency=kanna.settings.default_currency,
                           timezone_name=kanna.settings.timezone)


def _cmd_add(args: argparse.Namespace, kanna: Kanna) -> int:
    result = _service(kanna).add_transaction_from_text(args.text)
    print(result.message)
    return 0


def _cmd_query(args: argparse.Namespace, kanna: Kanna) -> int:
    result = _service(kanna).query_from_text(args.text)
    print(result.message)
    return 0


def _cmd_budget(args: argparse.Namespace, kanna: Kanna) -> int:
    svc = _service(kanna)
    from finance.money import Money
    from decimal import Decimal
    currency = args.currency or kanna.settings.default_currency
    money = Money.from_decimal(Decimal(str(args.amount)), currency)
    budget = svc.set_budget(category_name=args.category or None, amount_minor=money.amount_minor,
                             currency=currency, period=args.period, start_date=args.start_date,
                             end_date=args.end_date)
    print(f"Created budget {budget.id}: {money} ({args.period}, from {args.start_date})")
    return 0


def _cmd_recurring(args: argparse.Namespace, kanna: Kanna) -> int:
    svc = _service(kanna)
    from finance.money import Money
    from decimal import Decimal
    currency = args.currency or kanna.settings.default_currency
    money = Money.from_decimal(Decimal(str(args.amount)), currency)
    rec = svc.add_recurring(description=args.description, amount_minor=money.amount_minor,
                             currency=currency, frequency=args.frequency,
                             next_occurrence=args.next_occurrence, category_name=args.category or None)
    print(f"Created recurring expense {rec.id}: {money} {args.frequency}, next {args.next_occurrence}")
    return 0


def _cmd_import(args: argparse.Namespace, kanna: Kanna) -> int:
    svc = _service(kanna)
    try:
        with open(args.path, encoding="utf-8") as fh:
            csv_text = fh.read()
    except OSError as exc:
        print(f"error: could not read {args.path}: {exc}", file=sys.stderr)
        return 1
    result = import_csv(csv_text, tx_repo=svc.transactions, cat_repo=svc.categories,
                         default_currency=svc.default_currency)
    print(f"Imported {result.created} transaction(s), skipped {result.skipped_duplicates} duplicate(s), "
          f"{len(result.errors)} error(s).")
    for err in result.errors:
        print(f"  row {err.row_number}: {err.error}", file=sys.stderr)
    return 0


def _cmd_export(args: argparse.Namespace, kanna: Kanna) -> int:
    svc = _service(kanna)
    from finance.export import to_csv, to_json
    transactions = svc.search(start_date=args.start_date, end_date=args.end_date, limit=100_000)
    content = to_json(transactions) if args.format == "json" else to_csv(transactions)
    print(content)
    return 0
