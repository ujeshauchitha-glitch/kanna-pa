# Architecture

## Module map (Phase 1 + Phase 2)

```
core/
  config/       Settings (defaults → config.toml → KANNA_* env vars), filesystem paths
  logging/      Logging setup with automatic secret redaction
  events/       Synchronous pub/sub event bus (Event, EventBus)
  errors.py     The KannaError hierarchy every deliberate failure raises
  tools/        Tool protocol, Schema (validation + JSON-Schema export), ToolResult,
                ToolRegistry (the single choke point every tool call passes through), ToolContext
  permissions/  PermissionLevel, PermissionPolicy (rule-based decisions), ApprovalGate
                implementations, Sandbox (filesystem containment)
  memory/       Database (SQLite + migrations), one repository per concern
                (sessions, tasks, plans, execution_log)
  llm/          LLMProvider protocol, AnthropicProvider (real), NullProvider, FakeProvider (tests)
  planner/      Planner protocol, RuleBasedPlanner (deterministic), LLMPlanner (validates the
                model's plan against the tool registry before accepting it)
  agent/        AgentState, the AgentLoop itself, verifier (postcondition checks), AgentSession
  tasks/        The task-tracking system (distinct from finance and from plan execution)
  bootstrap.py  Wires every piece above into one Kanna object — the CLI and tests both use this

tools/
  filesystem/   read, write, list, search, mkdir, info, delete — all sandboxed
  process/      Controlled subprocess execution (allowlisted executables, no shell, timeout)
  computer/     ComputerAgent protocol, FedoraAgent (real, X11 via xdotool/scrot/xclip),
                NullComputerAgent (honest "unavailable"), FakeComputerAgent (tests),
                get_computer_agent() (backend selection), tools.py (registry-exposed computer_* tools)
  browser/      BrowserAgent protocol, PlaywrightBrowserAgent (real, Chromium-based),
                FakeBrowserAgent (tests), get_browser_agent() (process-level singleton session),
                tools.py (registry-exposed browser_* tools)
  scaffold/     Pure project-skeleton generators for C/C++/Java (templates.py — no ecosystem
                scaffolding tool exists for these, unlike cargo/npm), tools.py (project_scaffold)
  authoring/    ContentAuthor protocol, LLMAuthor (real — reads source material and tasks,
                produces Document-compatible structured content with provenance and unresolved
                questions), tools.py (registry-exposed author_content tool)

finance/        money.py (exact integer-minor-unit arithmetic), models.py, repository.py (SQL),
                dates.py (shared date-string normalization), nlp.py (deterministic NL parsing),
                analytics.py (the only place a total is computed), recurring.py (occurrence math),
                imports/ (csv_import.py, receipt.py, statement.py — both vision-backed,
                interfaces.py), export.py, service.py (orchestration), tools.py (registry-exposed
                finance_* tools)

vision/
  _common.py    Shared Anthropic-client + content-block helpers
  ocr/          OCRProvider protocol, AnthropicOCRProvider (real), FakeOCRProvider (tests)
  document/     DocumentProvider protocol, AnthropicDocumentProvider (real — receipt, multi-page
                statement, and generic structure extraction), FakeDocumentProvider (tests)
  tools.py      vision_extract_structure — the one vision capability registered directly (no
                finance-shaped consumer wraps it, unlike receipt/statement extraction)

documents/      model.py (shared Document/Section/TableData content model, format-independent),
                docx_writer.py / pptx_writer.py / pdf_writer.py (one real renderer each, offline —
                python-docx / python-pptx / reportlab), convert.py (DOCX/PPTX -> PDF via LibreOffice
                headless — real layout fidelity, needs soffice installed), tools.py
                (registry-exposed document_generate_*/document_convert_to_pdf tools, sandboxed +
                overwrite-gated like fs_write_file)

automation/
  scheduler/    Schedule (once/interval/weekly), pure due-time computation, SchedulerStore
                (SQLite), Scheduler.tick(), SchedulerDaemon (a real run-forever/bounded-ticks
                loop around tick() — see docs/SCHEDULER.md)

interfaces/
  cli/          argparse-based CLI; commands/ holds the larger per-area subcommand modules
                (finance, task, scheduler, document, computer, trust); app.py holds the smaller ones
                (init, ask, tools, db)

main.py         `python main.py <command> ...`
```

Directories present in the target structure but not yet populated (`runtimes/`, `work/`,
`education/`, `devices/`, `interfaces/voice`, `interfaces/desktop`, `interfaces/mobile`) are
deliberately not created until there's real code to put in them — see `docs/ROADMAP.md`.
`vision/` and `documents/` are populated as of Phase 2 — see `docs/VISION.md` and
`docs/DOCUMENTS.md`.

## The agent loop

Sequential plans now support backward result references (`core/planner/bindings.py`). Plan validation
checks literal structure and postconditions first; the loop resolves each step's arguments from
verified prior results before the unchanged registry validation/permission pipeline. Step persistence
includes resolved arguments and per-attempt raw results/verification errors. See `WORKFLOWS.md`.

```
UNDERSTAND → PLAN → SELECT TOOLS → EXECUTE → OBSERVE → CHECK → CORRECT IF NECESSARY → VERIFY → COMPLETE
```

`core/agent/loop.py` implements this as an explicit state machine (`core.agent.state.AgentState`).
"Understand"/"Plan"/"Select tools" collapse into one `Planner.create_plan()` call in Phase 1 — the
planner both interprets the request and picks tools + arguments. Each step is then:

1. **Execute** — `ToolRegistry.invoke()`.
2. **Observe** — the returned `ToolResult`.
3. **Check** — `core.agent.verifier.verify()` inspects the result against the step's declared
   `expected` postconditions **in code**, never by asking a model if it "looks right".
4. **Correct if necessary** — a failing step is retried (same tool) up to `max_corrections` times. If
   the planner supports it (`LLMPlanner.revise_step` — an *optional* capability `_run_step` detects
   with `hasattr`, not part of the `Planner` protocol itself, so `RuleBasedPlanner` needs no-op stub),
   the args are revised based on the specific failure reason before each retry; otherwise, or if that
   revision attempt itself fails (LLM unavailable, malformed response, invalid args), the retry uses
   the same args unchanged. Either way, the tool never changes mid-correction — only its args can.
5. **Verify** — re-run the check after each attempt. A failed postcondition is a failed step even
   when the tool itself returned success. Claimed output paths are checked inside the sandbox and
   process runs require exit code zero unless explicitly specified otherwise. Successful calls with
   failed checks are not replayed; denials, invalid input/output, and unavailable capabilities also
   terminate without retry. Other tool failures retain bounded argument correction.

If a step's postconditions never pass within the correction budget, the whole run reports **FAILED**
with the real blocker (the tool's error message or the specific postcondition that didn't hold) —
never a fabricated success. If the planner can't even produce a plan, the run reports **BLOCKED**.
Every transition is published on the event bus, and the plan + each step's outcome is persisted via
`PlanRepository` (`plans`/`plan_steps` tables), so a run is inspectable after the fact.

See `core/agent/loop.py`'s module docstring and `core/planner/llm_planner.py::LLMPlanner.revise_step`
for the correction design in full.

## The tool protocol

Every tool implements:

```python
class Tool(Protocol):
    name: str
    description: str
    input_schema: Schema
    output_schema: Schema
    permission: PermissionLevel
    def execute(self, args: dict, ctx: ToolContext) -> ToolResult: ...
```

`ToolRegistry.invoke()` is the only path anything (CLI, agent loop, LLM tool-use, a future scheduler
job) uses to call a tool. It always, in order: validates `args` against `input_schema` → asks
`PermissionPolicy.decide()` what to do → asks the `ApprovalGate` if approval is required → calls
`execute()` (catching any exception, so a buggy tool can never crash the agent) → validates the
result's `data` against `output_schema` on success → records the call in `execution_log` → publishes
a `tool.executed` event. The gate `bootstrap()` wires in is always a `TrustStoreGate` — it consults
persisted standing approvals (`kanna trust add/list/remove`) before falling back to `CLIPromptGate`/
`DenyAllGate`, so a trusted-automation grant applies to every entry point built on `bootstrap()`, not
just the session that granted it. See `docs/TOOLS.md` for the tool catalog and `docs/SECURITY.md` for
the permission model in detail.

## Persistence

SQLite via `core/memory/db.py`, with `PRAGMA foreign_keys = ON` and WAL mode for file-backed
databases. Migrations are plain numbered `.sql` files in `core/memory/migrations/`, applied through a
`schema_migrations` table — `Database.migrate()` is idempotent and safe to call on every startup.
Each concern gets its own table(s) rather than one shared blob: sessions/messages, tasks, plans/
plan_steps, execution_log, events, finance_* (categories, category_rules, transactions, budgets,
recurring), schedules/jobs, trust_rules (standing tool-approval grants). See the full schema in
`core/memory/migrations/0001_initial.sql` and `0002_trust_rules.sql`.

## Configuration

`core/config/settings.py` resolves settings in layers: built-in defaults → `config.toml`
(`KANNA_CONFIG_PATH`, default `~/.kanna/config.toml`) → `KANNA_*` environment variables. All of
Kanna's own state (database, logs, config) lives under `KANNA_HOME` (default `~/.kanna`), which also
defaults into the filesystem sandbox alongside the current working directory.

## Extending Kanna

Adding a new capability means: define its data model + a service layer that does the real work,
expose it as one or more `Tool` implementations with proper schemas and permission levels, register
those tools in `core/bootstrap.py`, and (if it should be reachable from natural language without an
LLM) teach `RuleBasedPlanner` to recognize a phrasing for it. Nothing in `core/agent`, `core/tools`,
or `interfaces/cli` needs to change.
