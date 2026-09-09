# Kanna — Specification

## What Kanna is

Kanna is a personal AI **work-execution agent**, not a chatbot. The design commits to one loop:

```
USER REQUEST → UNDERSTAND → PLAN → SELECT TOOLS → EXECUTE → OBSERVE → CHECK
    → CORRECT IF NECESSARY → VERIFY → COMPLETE
```

Wherever it's technically and safely possible, Kanna performs the work itself — reads and writes
files, runs code, queries its own database — rather than just describing steps for a human to do.

## Architectural principles

1. **Modular, not monolithic.** Every capability (filesystem, process execution, finance, planning,
   permissions...) is its own package behind a small interface (a `Protocol` or an ABC-shaped class).
   New capabilities register into the existing `ToolRegistry`/`Planner`/`AgentLoop` without changing
   their code.
2. **Tools are structured, not stringly-typed.** Every tool declares an input/output `Schema` and
   returns a `ToolResult` (`success`, `data`, `error`, `files_created/modified/deleted`, `metadata`,
   `duration_ms`) — never a bare string the caller has to parse.
3. **Deterministic where it matters.** The LLM interprets natural language into structured intents;
   plain code does every calculation, database write, and permission decision. This is enforced hard
   in `finance/` — see `docs/FINANCE.md`.
4. **Permission-gated by default.** Every tool has a `PermissionLevel`. LOW-risk actions (read, list,
   search, create-new-file, run allowlisted code) auto-run. REVIEW-risk actions (delete, overwrite,
   anything irreversible or external) require an `ApprovalGate` to say yes — the default gate says no
   to everything. See `docs/SECURITY.md`.
5. **Honest about what it can't do.** A capability with no real implementation reports
   `CapabilityUnavailable`/`VisionUnavailable`/`BrowserUnavailable` instead of a fake success — see
   `vision/ocr/anthropic_ocr.py` for the same discipline applied to a capability (vision) that *is*
   implemented but can still be unconfigured.
6. **Every run is inspectable.** The agent loop persists its plan, each step's result, and every tool
   invocation (`execution_log`) to SQLite, so a run can be audited after the fact.

## What Kanna can do today (Phase 1 + Phase 2)

- Take a natural-language request via `kanna ask "<request>"`, plan it (rule-based pattern matching,
  or an LLM planner when `ANTHROPIC_API_KEY` is set), execute it through the tool registry, verify
  each step's postconditions in code, retry a failing step up to a bounded correction budget, and
  report COMPLETE / FAILED (with the real blocker) / BLOCKED (when it can't even form a plan).
- Read, write (create or, with approval, overwrite), list, search, `mkdir`, inspect, and delete files
  — all sandboxed to configured roots (defaults: the current working directory + Kanna's own home).
- Run allowlisted interpreters/compilers (`python3`, `gcc`/`g++`, `rustc`, `javac`/`java`, `node`,
  `octave`, ...) as a real subprocess (no shell) with captured stdout/stderr/exit code/duration and a
  timeout.
- Track a personal finance ledger: natural-language transaction entry ("I spent ₹340 on lunch"),
  natural-language queries ("How much did I spend on food this month?"), categories with keyword-based
  auto-categorization, budgets with status/alerts, recurring expenses (with correct month-end
  rollover), CSV import (deduped), **receipt image/PDF import** and **bank/card statement import
  (multi-page PDF, debit rows only)** — both vision-backed — and CSV/JSON export. All arithmetic is
  exact integer minor-unit math — see `docs/FINANCE.md`.
- Read text out of an image or PDF (`vision/ocr`) and extract structured data — a receipt's merchant/
  date/amount/line items, every debit/credit row on a (possibly multi-page) bank/card statement, or an
  arbitrary document's title/sections/headings/paragraphs/bullets/tables (`vision/document`) — backed
  by Claude's vision capability. Requires `ANTHROPIC_API_KEY`; reports `VisionUnavailable` honestly
  otherwise. See `docs/VISION.md`.
- Generate DOCX, PPTX, and PDF files from structured content (title + sections with headings,
  paragraphs, bullets, tables) via `python-docx`/`python-pptx`/`reportlab` — fully offline,
  deterministic, no LLM or network involved in rendering itself. See `docs/DOCUMENTS.md`.
- Control the desktop on a Linux machine with a live X11 session: screenshot, mouse, keyboard,
  clipboard, open/close applications (`tools/computer/fedora.py::FedoraAgent`, `xdotool`/`scrot`/
  `xclip`-backed). Falls back to `NullComputerAgent` (honest `CapabilityUnavailable`, never a fake
  success) when no display or those binaries are detected. See `docs/DEVICES.md`.
- Drive a real Chromium browser: navigate, read page text, screenshot, click, and fill form fields
  (`tools/browser/playwright_backend.py::PlaywrightBrowserAgent`, Playwright-backed), with every
  state-changing action returning an observation of the resulting page — never a bare success flag.
  Reports `BrowserUnavailable` honestly if Playwright or a browser binary isn't installed. See
  `docs/BROWSER.md`.
- Track tasks (`kanna task add/list/start/complete/cancel`) and sessions/conversation history.
- Run scheduled jobs via `kanna scheduler tick` — one-time, interval, and "every N weeks on
  \<weekday\>" schedules, computed with pure, unit-tested date arithmetic.
- Persist everything to a local SQLite database with a real, idempotent migration system.

## What Kanna cannot do yet

- **Computer control on Windows or Phone, or device selection/routing across multiple devices** — the
  `ComputerAgent` interface has one real backend (`FedoraAgent`, X11-based — works on any Linux
  desktop with a live X11 session, not literally only Fedora) and one honest fallback
  (`NullComputerAgent`). No Windows/Phone backend, and no router across multiple registered devices,
  exist yet — see `docs/DEVICES.md`.
- **Handwriting generation, non-Python code runtimes beyond what's listed above** (Octave/C#/full Java
  toolchains depend on binaries that may not be installed on a given machine — the process tool will
  report that honestly rather than fake output).
- **Multi-tab/multi-context browsing, session persistence across restarts, file upload/download in the
  browser, or a `kanna browser ...` CLI subcommand** — the browser tool has one real backend
  (`PlaywrightBrowserAgent`) reachable only through the tool registry/agent loop today — see
  `docs/BROWSER.md`.
- **Offline/local OCR** — vision is real but Anthropic-only (no `tesseract`/local provider in this
  environment). Generic structure extraction reads visual structure, not document *semantics* — it
  can't tell "these are the instructions" from "this is a question" beyond what the source document's
  own headings convey, and there's no assignment-reading *workflow* built on top of it yet (that's
  future work — see `docs/ROADMAP.md`). Image editing/generation is also not built — see
  `docs/VISION.md`.
- **Income/credit tracking** — statement import only records debit (spend) rows; credit rows (deposits,
  refunds, salary) are reported, not imported, since `Transaction` has no signed-amount or income/
  expense representation — see `docs/FINANCE.md`.
- **Reading or editing existing DOCX/PPTX/PDF files, DOCX/PPTX→PDF conversion** — generation only, and
  only from structured content built by the caller (no "turn this rough idea into a full report"
  content-writing step; that's a job for an LLM *before* handing `documents` a `Document`) — see
  `docs/DOCUMENTS.md`.
- **Education/assignment workflows, multi-device orchestration.**
- **Trusted/pre-approved automations beyond `PreApprovedGate`'s explicit allowlist** — there is no UI
  yet for a user to grant standing approval; that's a policy-configuration feature for a later phase.

See `docs/ROADMAP.md` for what's planned next and in what order.

## Repository layout

See `docs/ARCHITECTURE.md` for the full module map and how the pieces fit together.
