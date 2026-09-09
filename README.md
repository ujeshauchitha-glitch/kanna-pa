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

`pip install -e ".[vision]"` + the same `ANTHROPIC_API_KEY` additionally enables reading receipts:

```bash
python main.py finance import-receipt path/to/receipt.jpg
```

## Project layout

```
core/          agent loop, planner, tool protocol/registry, permissions, memory (SQLite), config,
               logging, events, LLM provider abstraction, task system
tools/         filesystem, process execution, computer-control interface
finance/       transactions, categories, budgets, recurring expenses, import/export, analytics
vision/        OCR + receipt structure extraction, Anthropic-vision-backed
automation/    scheduler (once/interval/weekly schedules)
interfaces/    CLI
tests/         pytest suite, one file per subsystem
docs/          architecture, tools, finance, vision, devices, security, testing, roadmap
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full picture.
