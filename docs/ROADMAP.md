# Roadmap

Phase 1 built the foundation: agent loop, tool protocol/registry, permissions, SQLite persistence,
planner (rule-based + LLM), finance subsystem, filesystem/process tools, scheduler primitives, and a
CLI. Phase 2 added, in order: **vision** (OCR + receipt structure extraction, Anthropic-vision-backed,
plus the concrete `finance/imports/receipt.py` consumer left as an interface-only placeholder in
Phase 1), **document generation** (DOCX/PPTX/PDF from one shared content model, fully offline —
`python-docx`/`python-pptx`/`reportlab`, no LLM or network in the rendering path itself), a real
**`ComputerAgent` backend** (`FedoraAgent`, X11-based via `xdotool`/`scrot`/`xclip`, validated against
a live virtual display — see `docs/DEVICES.md`), **statement extraction** (multi-page bank/card
statement reading, extending `vision/document/` beyond single-receipt extraction, plus the concrete
`finance/imports/statement.py` consumer — the other interface-only placeholder from Phase 1), and a
**browser automation tool** (`PlaywrightBrowserAgent`, Chromium-based, validated against a real
headless browser — see `docs/BROWSER.md`), **generic document structure extraction**
(`extract_structure()`, extending `vision/document/` past the fixed receipt/statement-row shapes to
arbitrary sections/headings/paragraphs/tables, exposed as `vision_extract_structure` — see
`docs/VISION.md`), **LLM-driven correction** (`AgentLoop._run_step` now asks the planner to
revise a failing step's args based on the specific failure reason before retrying, when the planner
supports it — see `docs/ARCHITECTURE.md`'s agent-loop section), and **trusted-automation
configuration** (`TrustStoreGate`, `kanna trust add/list/remove` — persisted standing approval for
specific tool+argument patterns, shared across every entry point built on `bootstrap()` — see
`docs/SECURITY.md`), and a **scheduler daemon + `kanna scheduler add`** (`SchedulerDaemon` — a real
run-forever/bounded-ticks loop around `Scheduler.tick()`, a documented systemd unit — plus the CLI
command that was actually missing to create a schedule at all in Phase 1 — see `docs/SCHEDULER.md`),
**project scaffolding for C/C++/Java** (`project_scaffold`, plus `make` joining the `process_run`
allowlist so a generated Makefile-based project is actually buildable through Kanna's own tools, not
just generated — see `docs/RUNTIMES.md`), and **DOCX/PPTX → PDF conversion**
(`documents.convert.convert_to_pdf`, `document_convert_to_pdf`, `kanna document convert` — real
LibreOffice-headless conversion, found and fixed a missing-package gap where a minimal LibreOffice
install can't actually convert anything — see `docs/DOCUMENTS.md`). Everything below is not yet built.

## Vision (done, Phase 2) — what's left in this area

- No offline/local OCR provider (would need `tesseract` or similar; not installed in this
  environment, but `vision/ocr/base.py`'s `OCRProvider` Protocol means one can be added without
  touching callers).
- Generic structure extraction reads visual structure, not semantics — it can't yet tell "this
  section is the instructions" from "this section is a question" beyond what the source document's
  own headings already convey. An assignment-reading workflow that needs that distinction is still
  future work (see item 5 below).
- No `vision/image/` (general description/editing) or `vision/screen/` (screenshot understanding,
  tied to `tools/computer/` — now that a real `ComputerAgent` exists, this is more reachable than it
  was) yet.

## Document generation (done, Phase 2) — what's left in this area

- No reading or editing of existing DOCX/PPTX/PDF files — generation and DOCX/PPTX→PDF conversion
  only, though `document_read` now extracts structured content from DOCX files (sections, headings,
  bullets, tables).
- No page headers/footers, multi-column PDF layout, or embedded images.
- No PPTX theming beyond python-pptx's default template.
- No natural-language shortcut ("write me a report about X") — that's an LLM content-authoring step
  that has to happen *before* `documents`, which only renders content it's already given
  structured (see `docs/DOCUMENTS.md` for why this wasn't built into the rule-based planner).

## DOCX/PPTX → PDF conversion (done, this pass) — what's left in this area

- Depends on LibreOffice being installed — no pure-Python fallback (there isn't one worth trusting
  for real layout fidelity; see `docs/DOCUMENTS.md`).
- One direction only (DOCX/PPTX → PDF) — no PDF → DOCX or any other conversion.
- No batch/multi-file conversion — one `source`/`dest` pair per call.

## Computer control (done, Phase 2) — what's left in this area

- Two backends (`FedoraAgent` for X11/Linux, `WindowsAgent` for PowerShell/Windows) and one
  fallback (`NullComputerAgent`). No `PhoneAgent`, and no full router that picks a device by
  capability (`get_computer_agent()` checks Fedora → Windows → Null in order, not a selection
  across multiple *registered* devices).
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
  documents — that gap is now closed by generic structure extraction (see below).

## Generic document structure extraction (done, this pass) — what's left in this area

- Reads visual structure only, not document semantics — see the note in the Vision section above.
- No `kanna vision ...` CLI subcommand yet (unlike `finance import-receipt`) — registry/agent-loop
  path only.
- No mapping helper from `vision.document.base.DocumentStructure` to `documents.model.Document` yet
  (deliberately kept out of `vision`, which has no dependency on `documents` — see `docs/VISION.md`);
  a caller wanting to re-render an extracted document builds that mapping itself for now.

## Browser automation (done, this pass) — what's left in this area

- One page at a time, one process-level session (`get_browser_agent()`) — no multi-tab/multi-context
  support, and no cookie/session persistence across process restarts.
- `browser_get_text`/`text_excerpt` is whole-page `inner_text("body")`, truncated — no
  selector-scoped text reading, no structured DOM/table extraction.
- No file-upload or file-download handling.
- No `kanna browser ...` CLI subcommand yet (unlike `computer`/`document`) — registry/agent-loop path
  only. See `docs/BROWSER.md`.

## LLM-driven correction (done, this pass) — what's left in this area

- `revise_step()` fixes one step's *args* based on the failure reason; it can't swap to a different
  tool mid-correction, retry a whole sub-sequence of steps, or use anything beyond the single most
  recent failure (no memory of earlier attempts within the same step, no cross-step learning within a
  run).
- No real-world failure corpus yet to know how well this actually helps versus a blind retry — it's
  built and tested against scripted `FakeProvider` responses, not evaluated against live failures.
- `RuleBasedPlanner` has no LLM to ask, so it still retries identically — this only helps runs using
  `LLMPlanner`.

## Trusted-automation configuration (done, this pass) — what's left in this area

- Matching is exact-value-per-key only — no wildcards, ranges, or path-prefix matching (e.g. "trust
  `fs_delete` under `~/scratch/` specifically" isn't expressible as one rule yet; it needs one rule per
  exact path).
- No expiry, scoping to a session/device, or interactive "approve once, and offer to remember this"
  flow — a grant is permanent until `kanna trust remove`.
- No `kanna trust` output redaction — an args pattern containing something sensitive would show up in
  `kanna trust list` verbatim (unlikely in practice, since REVIEW-level tools' args are file paths,
  selectors, coordinates, and app names, not secrets, but not defended against either).

## Scheduler daemon / OS integration (done, this pass) — what's left in this area

- One process per daemon instance — no distributed/multi-worker coordination.
- The systemd unit in `docs/SCHEDULER.md` is documentation to copy and adapt, not something any
  Kanna command installs.
- `kanna scheduler add` doesn't validate an ISO date/time string's *syntax* up front — a malformed
  `--run-at`/`--anchor-date` fails later as a Python exception rather than a clean CLI error.
- No pause/resume — `remove` is a one-way deactivation; there's no CLI command to reactivate a
  removed schedule.

## Runtimes beyond Python (done, this pass) — what's left in this area

- No dependency resolution — every generated skeleton has zero external dependencies (no
  vcpkg/conan for C/C++, no Maven/Gradle for Java).
- No `kanna scaffold ...` CLI subcommand yet — registry/agent-loop path only.
- No test-framework scaffolding, no library-vs-binary distinction, no MSVC/`nmake` support.

## Next candidates, roughly in order of leverage

The sequential workflow foundation is implemented: typed backward result references, runtime schema
checks, sandboxed artifact existence verification, process exit checks, per-attempt history, actual
result messages, and honest scheduled outcome propagation (`WORKFLOWS.md`). Source-aware content
authoring (`author_content`) now bridges source reading and document generation with LLM-driven
content creation, provenance tracking, and explicit unresolved questions. `document_read` extracts
structured content from DOCX files, and `assignment_solve` provides end-to-end question extraction
and answering with source-grounded answers. `WindowsAgent` provides real desktop automation on
Windows via PowerShell/.NET. The desktop task workspace now has a real single-instance guard, a
cross-platform global hotkey (Windows + X11 Linux), a task history browser over the agent loop's own
persisted plan/step evidence (no new schema, no rerun), cooperative cancellation (`AgentLoop.run`
checks a `threading.Event` before the next step/retry, never mid-call; both LLM providers now carry
a real request timeout so a hung model server can't defeat it), per-user startup-at-login
(Windows `HKCU` Run key / Linux XDG autostart, explicit Enable/Disable, never automatic), and an
editable connection-settings form (provider/model/fallback/timeout, writing to `config.toml` via a
narrow scalar-line writer, `core.config.settings.update_config_file`) — see `docs/DESKTOP.md`.
**The desktop-access milestone from the original brief is now complete**, including the connection
setup forms called out as a follow-up.

1. **Phone/device backend** and the full capability-based router across all registered devices, now
   that Fedora and Windows backends have validated the `ComputerAgent` interface in practice — the
   next-highest-leverage item, and the only one left from the original brief's device-backend list.
   Needs real hardware or an emulator to validate against before it can be built to the same bar
   every other capability here was held to; none is available in this build environment yet.

## Explicitly deferred, no strong opinion yet

- Desktop and voice are implemented. The desktop workspace now supports serialized execution,
  source selection, approvals, output files, voice-to-draft, a real single-instance guard, a real
  cross-platform global hotkey (Windows + X11 Linux; Wayland and macOS honestly report unavailable
  rather than faking it), a read-only task history browser (`core/agent/history.py` +
  `interfaces/desktop/history_dialog.py`), cooperative cancellation, per-user startup-at-login
  (`interfaces/desktop/startup.py` + `startup_dialog.py`), and an editable connection settings form
  (`interfaces/desktop/connection_dialog.py`) — see `docs/DESKTOP.md`. Mobile UI is deferred.
- Multi-device task routing beyond the capability-matching sketch in `docs/DEVICES.md`.
