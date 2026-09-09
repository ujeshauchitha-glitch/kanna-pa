"""Finance capabilities exposed through the tool registry.

Every tool here is a thin wrapper around `FinanceService` — no tool does
its own arithmetic or database access, so the deterministic guarantee in
`finance/service.py` holds no matter which path (CLI, agent loop, LLM
tool-use) reaches these.
"""
from __future__ import annotations

from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.registry import ToolRegistry
from core.tools.result import ToolResult
from core.tools.schema import array, integer, obj, string
from finance.export import to_csv, to_json
from finance.imports.csv_import import import_csv
from finance.repository import CategoryRepository, TransactionRepository
from finance.service import FinanceService


def _service(ctx: ToolContext) -> FinanceService:
    return FinanceService(ctx.db, default_currency=ctx.settings.default_currency,
                           timezone_name=ctx.settings.timezone)


class FinanceAddTransactionTool:
    name = "finance_add_transaction"
    description = "Record a spending transaction from a natural-language sentence, e.g. 'I spent ₹340 on lunch'."
    permission = PermissionLevel.LOW
    input_schema = obj({"text": string(description="Natural-language description of the spend")},
                        required=("text",))
    output_schema = obj({
        "transaction_id": string(), "amount_minor": integer(), "currency": string(),
        "occurred_at": string(), "category": string(), "message": string(),
    })

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        result = _service(ctx).add_transaction_from_text(args["text"])
        tx = result.transaction
        return ToolResult.ok({
            "transaction_id": tx.id, "amount_minor": tx.amount_minor, "currency": tx.currency,
            "occurred_at": tx.occurred_at, "category": result.category_name or "",
            "message": result.message,
        })


class FinanceQueryTool:
    name = "finance_query"
    description = "Answer a natural-language spending question, e.g. 'How much did I spend on food this month?'."
    permission = PermissionLevel.LOW
    input_schema = obj({"text": string(description="Natural-language spending question")},
                        required=("text",))
    output_schema = obj({
        "message": string(), "start_date": string(), "end_date": string(),
        "totals": array(obj({"currency": string(), "amount_minor": integer()})),
        "transaction_count": integer(),
    })

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        result = _service(ctx).query_from_text(args["text"])
        totals = [{"currency": m.currency, "amount_minor": m.amount_minor}
                   for m in result.summary.totals_by_currency.values()]
        return ToolResult.ok({
            "message": result.message, "start_date": result.summary.start_date,
            "end_date": result.summary.end_date, "totals": totals,
            "transaction_count": result.summary.transaction_count,
        })


class FinanceSetBudgetTool:
    name = "finance_set_budget"
    description = "Set a spending budget for a category (or overall) for a period."
    permission = PermissionLevel.LOW
    input_schema = obj(
        {
            "category": string(description="Category name, or empty for an overall budget", default=""),
            "amount_minor": integer(description="Budget amount in minor units (e.g. paise)"),
            "currency": string(description="ISO currency code"),
            "period": string(description="monthly | weekly | yearly | custom"),
            "start_date": string(description="YYYY-MM-DD"),
            "end_date": string(description="YYYY-MM-DD, optional", default=""),
        },
        required=("amount_minor", "currency", "period", "start_date"),
    )
    output_schema = obj({"budget_id": string()})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        budget = _service(ctx).set_budget(
            category_name=args.get("category") or None, amount_minor=args["amount_minor"],
            currency=args["currency"], period=args["period"], start_date=args["start_date"],
            end_date=args.get("end_date") or None,
        )
        return ToolResult.ok({"budget_id": budget.id})


class FinanceAddRecurringTool:
    name = "finance_add_recurring"
    description = "Register a recurring expense (subscription, rent, etc.)."
    permission = PermissionLevel.LOW
    input_schema = obj(
        {
            "description": string(),
            "amount_minor": integer(),
            "currency": string(),
            "frequency": string(description="daily | weekly | biweekly | monthly | yearly"),
            "next_occurrence": string(description="YYYY-MM-DD"),
            "category": string(default=""),
        },
        required=("description", "amount_minor", "currency", "frequency", "next_occurrence"),
    )
    output_schema = obj({"recurring_id": string()})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        rec = _service(ctx).add_recurring(
            description=args["description"], amount_minor=args["amount_minor"],
            currency=args["currency"], frequency=args["frequency"],
            next_occurrence=args["next_occurrence"], category_name=args.get("category") or None,
        )
        return ToolResult.ok({"recurring_id": rec.id})


class FinanceImportCsvTool:
    name = "finance_import_csv"
    description = "Import transactions from CSV text (columns: date, amount, description, merchant, category)."
    permission = PermissionLevel.LOW
    input_schema = obj({"csv_text": string(description="Raw CSV content")}, required=("csv_text",))
    output_schema = obj({"created": integer(), "skipped_duplicates": integer(), "error_count": integer()})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        svc = _service(ctx)
        result = import_csv(args["csv_text"], tx_repo=svc.transactions, cat_repo=svc.categories,
                              default_currency=svc.default_currency)
        return ToolResult.ok({
            "created": result.created, "skipped_duplicates": result.skipped_duplicates,
            "error_count": len(result.errors),
        }, metadata={"errors": [{"row": e.row_number, "error": e.error} for e in result.errors]})


class FinanceExportTool:
    name = "finance_export"
    description = "Export transactions matching a date range as CSV or JSON."
    permission = PermissionLevel.LOW
    input_schema = obj(
        {
            "start_date": string(default=""),
            "end_date": string(default=""),
            "format": string(enum=("csv", "json"), default="csv"),
        },
    )
    output_schema = obj({"content": string(), "count": integer()})

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        svc = _service(ctx)
        transactions = svc.search(start_date=args.get("start_date") or None,
                                   end_date=args.get("end_date") or None, limit=100_000)
        fmt = args.get("format", "csv")
        content = to_json(transactions) if fmt == "json" else to_csv(transactions)
        return ToolResult.ok({"content": content, "count": len(transactions)})


ALL_TOOLS = [
    FinanceAddTransactionTool(), FinanceQueryTool(), FinanceSetBudgetTool(),
    FinanceAddRecurringTool(), FinanceImportCsvTool(), FinanceExportTool(),
]


def register_all(registry: ToolRegistry) -> None:
    for tool in ALL_TOOLS:
        registry.register(tool)
