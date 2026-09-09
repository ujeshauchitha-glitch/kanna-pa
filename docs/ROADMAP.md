# Roadmap

Phase 1 built the foundation: agent loop, tool protocol/registry, permissions, SQLite persistence,
planner (rule-based + LLM), finance subsystem, filesystem/process tools, scheduler primitives, and a
CLI. Phase 2 (this pass) added vision: OCR + receipt structure extraction, both Anthropic-vision-
backed, plus the concrete `finance/imports/receipt.py` consumer that was left as an interface-only
placeholder in Phase 1. Everything below is not yet built.

## Vision (done, Phase 2) — what's left in this area

- No offline/local OCR provider (would need `tesseract` or similar; not installed in this
  environment, but `vision/ocr/base.py`'s `OCRProvider` Protocol means one can be added without
  touching callers).
- Only receipt extraction is implemented under `vision/document/`. Generic multi-page document
  structure extraction (needed for PDF assignment reading and bank statement import) is a distinct,
  larger piece of work — see item 3 below.
- No `vision/image/` (general description/editing) or `vision/screen/` (screenshot understanding,
  tied to `tools/computer/`) yet.

## Next candidates, roughly in order of leverage

1. **Document generation (`work/`, `docs` extra).** DOCX/PDF/PPTX output for reports and assignments,
   using `python-docx`/`pypdf`-style libraries behind their own tool(s), with the same
   execute-then-verify discipline the agent loop already applies (a generated file's existence and
   non-empty size is a trivial, real postcondition).
2. **One real `ComputerAgent` backend** (Fedora first, since that's the primary dev environment) —
   proves out the device-selection design named in `docs/DEVICES.md` before building the others.
3. **Generic document structure extraction.** Extend `vision/document/` beyond single-receipt
   extraction to multi-page/multi-section documents — unlocks PDF assignment reading (extract
   questions, instructions, reference material) and bank statement import (`StatementImporter` in
   `finance/imports/interfaces.py`, still unimplemented). Likely a new `extract_structure()` method
   alongside `extract_receipt()`, since the receipt-specific JSON shape doesn't generalize cleanly.
4. **Browser tool.** Navigation, page reading, form interaction, screenshots — critically, every
   action must be followed by an observation step (never assume a click succeeded), matching the
   agent loop's existing verify-after-execute pattern.
5. **LLM-driven correction.** Right now a failing step retries identically. Once there's a real
   failure corpus to learn from, make `AgentLoop._run_step` ask the planner to revise a step's args
   based on the specific failure reason before retrying — still bounded by `max_corrections`, still
   verified in code afterward.
6. **Trusted-automation configuration.** `PreApprovedGate` already supports it structurally; needs a
   config surface (CLI or file) for the user to grant standing approval to specific tool+argument
   patterns, plus an audit trail of what's been pre-approved.
7. **Scheduler daemon / OS integration.** `kanna scheduler tick` works today invoked manually or via
   cron/systemd-timer; a longer-running daemon mode (or documented systemd unit) makes it actually
   "set and forget."
8. **Runtimes beyond Python.** C/C++/Rust/Java toolchains are already reachable through
   `process_run`'s allowlist when installed; what's missing is per-language project scaffolding
   (build file generation, dependency resolution) if Kanna should set those up itself rather than
   just compile/run what's already there.
9. **Education/assignment workflow, NeoColab integration, handwriting rendering.** These depend on
   #1 and #3 above being solid first.
10. **Additional device backends** (Windows, Phone) once the Fedora backend has validated the
    `ComputerAgent` interface in practice.

## Explicitly deferred, no strong opinion yet

- Voice interface, desktop/mobile UI shells (`interfaces/voice`, `interfaces/desktop`,
  `interfaces/mobile`) — CLI is the only interface until there's a concrete reason to add another.
- Multi-device task routing beyond the capability-matching sketch in `docs/DEVICES.md`.
