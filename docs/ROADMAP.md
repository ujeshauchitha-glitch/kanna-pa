# Roadmap

Phase 1 built the foundation: agent loop, tool protocol/registry, permissions, SQLite persistence,
planner (rule-based + LLM), finance subsystem, filesystem/process tools, scheduler primitives, and a
CLI. Phase 2 added, in order: **vision** (OCR + receipt structure extraction, Anthropic-vision-backed,
plus the concrete `finance/imports/receipt.py` consumer left as an interface-only placeholder in
Phase 1), **document generation** (DOCX/PPTX/PDF from one shared content model, fully offline —
`python-docx`/`python-pptx`/`reportlab`, no LLM or network in the rendering path itself), a real
**`ComputerAgent` backend** (`FedoraAgent`, X11-based via `xdotool`/`scrot`/`xclip`, validated against
a live virtual display — see `docs/DEVICES.md`), and **statement extraction** (multi-page bank/card
statement reading, extending `vision/document/` beyond single-receipt extraction, plus the concrete
`finance/imports/statement.py` consumer — the other interface-only placeholder from Phase 1).
Everything below is not yet built.

## Vision (done, Phase 2) — what's left in this area

- No offline/local OCR provider (would need `tesseract` or similar; not installed in this
  environment, but `vision/ocr/base.py`'s `OCRProvider` Protocol means one can be added without
  touching callers).
- Receipt and statement extraction only — no *generic* document structure extraction (arbitrary
  sections/headings/tables, not a fixed receipt or transaction-row shape) yet. That's what PDF
  assignment reading (extract questions, instructions, reference material) still needs — see item 1
  below.
- No `vision/image/` (general description/editing) or `vision/screen/` (screenshot understanding,
  tied to `tools/computer/` — now that a real `ComputerAgent` exists, this is more reachable than it
  was) yet.

## Document generation (done, Phase 2) — what's left in this area

- No DOCX/PPTX → PDF conversion (LibreOffice/`soffice` is present in this environment and could back
  one via the existing controlled-process-execution pattern, but wasn't built — direct `reportlab`
  PDF generation was chosen instead so PDF output doesn't depend on what's installed on a given host).
- No reading or editing of existing DOCX/PPTX/PDF files — generation only.
- No page headers/footers, multi-column PDF layout, or embedded images.
- No PPTX theming beyond python-pptx's default template.
- No natural-language shortcut ("write me a report about X") — that's an LLM content-authoring step
  that has to happen *before* `documents`, which only renders content it's already given
  structured (see `docs/DOCUMENTS.md` for why this wasn't built into the rule-based planner).

## Computer control (done, Phase 2) — what's left in this area

- One backend (`FedoraAgent`, X11) and one fallback (`NullComputerAgent`) — no `WindowsAgent`/
  `PhoneAgent`, and no router that picks a device by capability (`get_computer_agent()` is a fixed
  choice between the two, not a selection across multiple *registered* devices).
- Wayland desktops without XWayland aren't supported (`xdotool`/`scrot`/`xclip` are X11 tools) — most
  Wayland compositors, including Fedora's default GNOME session, do run XWayland, so this covers more
  than it sounds like, but a native-Wayland backend (`wtype`/`grim`/`wl-clipboard`) isn't built.
- No window-content inspection beyond "what's the active window's title" (`inspect_screen()`) — e.g.
  no accessibility-tree reading, no OCR-the-screenshot-to-find-a-button. Combining `computer_screenshot`
  with `vision/ocr` for that is possible today (both are real tools) but no tool/workflow wires them
  together yet.

## Statement extraction (done, this pass) — what's left in this area

- Debit-only — no income/credit tracking, since `Transaction` has no signed-amount or income/expense
  representation. Credit rows are reported (visible, not silently dropped), not imported — see
  `finance/imports/statement.py`'s docstring and `docs/FINANCE.md` for the reasoning and where that
  boundary would move if Kanna grows income tracking later.
- The statement-specific JSON shape (fixed transaction-row fields) doesn't generalize to arbitrary
  documents — this doesn't unlock PDF assignment reading, which needs genuinely generic structure
  extraction instead (see item 1 below).

## Next candidates, roughly in order of leverage

1. **Generic document structure extraction.** Extend `vision/document/` with a third shape — arbitrary
   sections/headings/paragraphs/tables, not a fixed receipt or statement-row schema — to unlock PDF
   assignment reading (extract questions, instructions, reference material). Likely a new
   `extract_structure()` method alongside `extract_receipt()`/`extract_statement()`.
2. **Browser tool.** Navigation, page reading, form interaction, screenshots — critically, every
   action must be followed by an observation step (never assume a click succeeded), matching the
   agent loop's existing verify-after-execute pattern. Playwright is already available in this
   environment (used for testing, not yet wired into a Kanna tool).
3. **LLM-driven correction.** Right now a failing step retries identically. Once there's a real
   failure corpus to learn from, make `AgentLoop._run_step` ask the planner to revise a step's args
   based on the specific failure reason before retrying — still bounded by `max_corrections`, still
   verified in code afterward.
4. **Trusted-automation configuration.** `PreApprovedGate` already supports it structurally; needs a
   config surface (CLI or file) for the user to grant standing approval to specific tool+argument
   patterns, plus an audit trail of what's been pre-approved. More valuable now that `computer_click`/
   `computer_type_text` exist and are REVIEW-gated by default.
5. **Scheduler daemon / OS integration.** `kanna scheduler tick` works today invoked manually or via
   cron/systemd-timer; a longer-running daemon mode (or documented systemd unit) makes it actually
   "set and forget."
6. **Runtimes beyond Python.** C/C++/Rust/Java toolchains are already reachable through
   `process_run`'s allowlist when installed; what's missing is per-language project scaffolding
   (build file generation, dependency resolution) if Kanna should set those up itself rather than
   just compile/run what's already there.
7. **Education/assignment workflow, NeoColab integration, handwriting rendering.** These depend on
   document generation (done) and #1 above (generic document extraction) being solid first — an
   assignment workflow is essentially "read the assignment PDF via vision, do the work, write it up
   via `documents`."
8. **Additional device backends** (Windows, Phone) and the capability-based router across them, now
   that `FedoraAgent` has validated the `ComputerAgent` interface in practice.

## Explicitly deferred, no strong opinion yet

- Voice interface, desktop/mobile UI shells (`interfaces/voice`, `interfaces/desktop`,
  `interfaces/mobile`) — CLI is the only interface until there's a concrete reason to add another.
- Multi-device task routing beyond the capability-matching sketch in `docs/DEVICES.md`.
