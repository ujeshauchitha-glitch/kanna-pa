# Kanna

Kanna is a personal AI work-execution agent — not a chatbot. It understands a request, plans the
necessary steps, selects and uses tools, executes work, observes the results, corrects failures, and
reports a completed result.

```
USER REQUEST → UNDERSTAND → PLAN → SELECT TOOLS → EXECUTE → OBSERVE → CHECK
    → CORRECT IF NECESSARY → VERIFY → COMPLETE
```

See **[`KANNA_SPEC.md`](KANNA_SPEC.md)** for what Kanna is, what it can do today, and what it can't
do yet. See **[`docs/`](docs/)** for architecture, the tool catalog, the finance subsystem, device
plans, security model, testing, and the roadmap.

## Quick start

```bash
pip install -e .          # core has zero third-party dependencies
pip install -e ".[dev]"   # + pytest, for running the test suite
python -m pytest -q

python main.py init
python main.py finance add "I spent ₹340 on lunch"
python main.py finance query "How much did I spend on food this month?"
python main.py ask "list the files in ./docs"
python main.py tools list
```

Optionally, `pip install -e ".[llm]"` and `export ANTHROPIC_API_KEY=...` to enable the LLM-backed
planner for open-ended requests beyond what the built-in rule-based planner recognizes. Kanna runs
fully offline without it — the rule-based planner and the finance NLP parser never need an LLM.

`pip install -e ".[vision]"` + the same `ANTHROPIC_API_KEY` additionally enables reading receipts and
bank/card statements (multi-page PDFs supported; debit rows only — see `docs/FINANCE.md`):

```bash
python main.py finance import-receipt path/to/receipt.jpg
python main.py finance import-statement path/to/statement.pdf
```

The same install also enables reading an arbitrary document's structure (title, sections, headings,
paragraphs, tables) via the `vision_extract_structure` tool — reachable through the tool
registry/agent loop today, no CLI subcommand yet.

`pip install -e ".[documents]"` enables generating DOCX/PPTX/PDF files — fully offline, no API key:

```bash
python main.py document generate content.json --format pdf   # or docx, pptx
```

With LibreOffice installed (`libreoffice-writer`/`libreoffice-impress` — `libreoffice-core` alone
isn't enough, see `docs/DOCUMENTS.md`), convert an existing DOCX/PPTX to PDF:

```bash
python main.py document convert report.docx report.pdf
```

On a Linux machine with a live X11 session and `xdotool`/`scrot`/`xclip` installed
(`dnf install xdotool scrot xclip` on Fedora), computer control works out of the box — no extra
`pip install`:

```bash
python main.py computer screenshot shot.png
python main.py computer click 100 200
```

`pip install -e ".[browser]"` + `playwright install chromium` enables browser automation (navigate,
read page text, screenshot, click, fill) via `browser_*` tools — reachable through the tool
registry/agent loop today (no `kanna browser ...` CLI subcommand yet). See `docs/BROWSER.md`.

A REVIEW-level action (delete, click, overwrite, ...) normally asks for approval every time. Grant it
standing approval instead:

```bash
python main.py trust add computer_open_application --arg name=firefox --note "safe to auto-launch"
python main.py trust list
python main.py trust remove 1
```

See "Approval gates" in `docs/SECURITY.md`.

Create and run scheduled jobs — once via cron/systemd-timer, or continuously with a real daemon:

```bash
python main.py scheduler add "morning briefing" "How much did I spend on food this month?" \
    --kind weekly --weekday 0 --time 08:00 --anchor-date 2024-01-01
python main.py scheduler list
python main.py scheduler tick              # run whatever's due, once, and exit
python main.py scheduler daemon            # run forever, checking every 60s (Ctrl-C to stop)
```

See `docs/SCHEDULER.md`, including a documented systemd unit for running the daemon as a service.

`project_scaffold` generates a minimal, real, buildable C/C++/Java project skeleton — reachable
through the tool registry/agent loop today, no CLI subcommand yet (for Rust or Node, use
`process_run` with `cargo new`/`npm init` instead, since those already have a real scaffolding tool).
See `docs/RUNTIMES.md`.

## Project layout

Launch the native task workspace with `python main.py app`: multiline requests, source-file
selection, progress, approval dialogs, and verified output paths. Voice input fills a draft for
review. See [desktop setup and behavior](docs/DESKTOP.md) for provider configuration and limitations.

```
core/          agent loop, planner, tool protocol/registry, permissions, memory (SQLite), config,
               logging, events, LLM provider abstraction, task system
tools/         filesystem, process execution (incl. make), computer control (FedoraAgent — real,
               X11-based), browser automation (PlaywrightBrowserAgent — real, Chromium-based),
               project scaffolding for C/C++/Java
finance/       transactions, categories, budgets, recurring expenses, import/export, analytics
vision/        OCR + receipt/statement structure extraction, Anthropic-vision-backed
documents/     DOCX/PPTX/PDF generation from one shared content model (fully offline), plus
               DOCX/PPTX -> PDF conversion via LibreOffice headless
automation/    scheduler (once/interval/weekly schedules) + a real daemon mode
interfaces/    CLI
tests/         pytest suite, one file per subsystem
docs/          architecture, tools, finance, vision, documents, devices, browser, scheduler,
               runtimes, security, testing, roadmap
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full picture.
