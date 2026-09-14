"""Finance capabilities exposed through the tool registry.

Every tool here is a thin wrapper around `FinanceService` — no tool does
its own arithmetic or database access, so the deterministic guarantee in
`finance/service.py` holds no matter which path (CLI, agent loop, LLM
tool-use) reaches these.
"""
from __future__ import annotations

from core.errors import SandboxViolation
from core.permissions.levels import PermissionLevel
from core.tools.context import ToolContext
from core.tools.registry import ToolRegistry
from core.tools.result import ToolResult
from core.tools.schema import array, integer, obj, string
from finance.export import to_csv, to_json
from finance.imports.csv_import import import_csv
from finance.imports.receipt import import_receipt
from finance.imports.statement import import_statement
from finance.service import FinanceService
from vision._common import guess_mime_type
from vision.document.anthropic_document import AnthropicDocumentProvider
from vision.document.base import DocumentProvider


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


class FinanceImportReceiptTool:
    """Logs a transaction from a photographed/scanned receipt (image or PDF).

    Vision-backed extraction (see `vision.document`) is used only to
    *read* what's printed on the receipt — the amount is then parsed
    and persisted the same deterministic way as every other entry path
    (`finance/imports/receipt.py`).
    """

    name = "finance_import_receipt"
    description = "Log a transaction by reading a receipt image or PDF (path within the sandbox)."
    permission = PermissionLevel.LOW
    input_schema = obj(
        {
            "path": string(description="Path to the receipt image/PDF, within the sandbox"),
            "mime_type": string(description="Override the guessed mime type, e.g. 'image/png'",
                                 default=""),
        },
        required=("path",),
    )
    output_schema = obj({
        "created": integer(), "skipped_duplicate": string(), "transaction_id": string(),
        "error": string(),
    })

    def __init__(self, provider: DocumentProvider | None = None) -> None:
        self._provider = provider

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            resolved = ctx.sandbox.resolve(args["path"])
        except SandboxViolation as exc:
            return ToolResult.fail("sandbox_violation", str(exc))

        if not resolved.exists() or not resolved.is_file():
            return ToolResult.fail("not_found", f"file does not exist: {resolved}")

        mime_type = args.get("mime_type") or ""
        if not mime_type:
            try:
                mime_type = guess_mime_type(resolved)
            except ValueError as exc:
                return ToolResult.fail("unsupported_file_type", str(exc))

        provider = self._provider or AnthropicDocumentProvider(
            model=ctx.settings.llm_model, max_tokens=ctx.settings.llm_max_tokens,
        )
        svc = _service(ctx)
        result = import_receipt(
            resolved.read_bytes(), mime_type=mime_type, tx_repo=svc.transactions,
            cat_repo=svc.categories, provider=provider, default_currency=svc.default_currency,
        )

        if result.errors:
            return ToolResult.fail("receipt_extraction_failed", result.errors[0].error)
        if result.skipped_duplicates:
            return ToolResult.ok({"created": 0, "skipped_duplicate": "True", "transaction_id": "",
                                   "error": ""})
        return ToolResult.ok({"created": result.created, "skipped_duplicate": "False",
                               "transaction_id": result.created_transaction_ids[0], "error": ""})


class FinanceImportStatementTool:
    """Imports spend (debit) transactions from a bank/card statement image or PDF.

    Same vision-reads-only-finance-parses discipline as receipt import
    (`finance/imports/statement.py`). Unlike receipt import (exactly one
    expected transaction, so any error is total failure), a statement's
    rows are reported with the same tolerant partial-success shape as
    `finance_import_csv` — a skipped credit row or one unparseable row
    doesn't fail the whole import.
    """

    name = "finance_import_statement"
    description = ("Import spend (debit) transactions from a bank/card statement image or PDF "
                    "(path within the sandbox, may be multi-page). Credit rows are reported, not "
                    "imported — see docs/FINANCE.md.")
    permission = PermissionLevel.LOW
    input_schema = obj(
        {
            "path": string(description="Path to the statement image/PDF, within the sandbox"),
            "mime_type": string(description="Override the guessed mime type, e.g. 'application/pdf'",
                                 default=""),
        },
        required=("path",),
    )
    output_schema = obj({"created": integer(), "skipped_duplicates": integer(), "error_count": integer()})

    def __init__(self, provider: DocumentProvider | None = None) -> None:
        self._provider = provider

    def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            resolved = ctx.sandbox.resolve(args["path"])
        except SandboxViolation as exc:
            return ToolResult.fail("sandbox_violation", str(exc))

        if not resolved.exists() or not resolved.is_file():
            return ToolResult.fail("not_found", f"file does not exist: {resolved}")

        mime_type = args.get("mime_type") or ""
        if not mime_type:
            try:
                mime_type = guess_mime_type(resolved)
            except ValueError as exc:
                return ToolResult.fail("unsupported_file_type", str(exc))

        provider = self._provider or AnthropicDocumentProvider(
            model=ctx.settings.llm_model, max_tokens=ctx.settings.llm_max_tokens,
        )
        svc = _service(ctx)
        result = import_statement(
            resolved.read_bytes(), mime_type=mime_type, tx_repo=svc.transactions,
            cat_repo=svc.categories, provider=provider, default_currency=svc.default_currency,
        )

        # row_number == 0 is this module's convention for a whole-document
        # failure (provider error, no transactions found at all) rather
        # than a per-row issue — only that case fails the tool call.
        document_level_error = next((e for e in result.errors if e.row_number == 0), None)
        if document_level_error and result.created == 0 and result.skipped_duplicates == 0:
            return ToolResult.fail("statement_extraction_failed", document_level_error.error)

        return ToolResult.ok({
            "created": result.created, "skipped_duplicates": result.skipped_duplicates,
            "error_count": len(result.errors),
        }, metadata={"errors": [{"row": e.row_number, "error": e.error} for e in result.errors]})


ALL_TOOLS = [
    FinanceAddTransactionTool(), FinanceQueryTool(), FinanceSetBudgetTool(),
    FinanceAddRecurringTool(), FinanceImportCsvTool(), FinanceExportTool(),
    FinanceImportReceiptTool(), FinanceImportStatementTool(),
]


def register_all(registry: ToolRegistry) -> None:
    for tool in ALL_TOOLS:
        registry.register(tool)
