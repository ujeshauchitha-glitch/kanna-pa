# Roadmap

Phase 1 (this pass) built the foundation: agent loop, tool protocol/registry, permissions, SQLite
persistence, planner (rule-based + LLM), finance subsystem, filesystem/process tools, scheduler
primitives, and a CLI. Everything below is not yet built.

## Phase 2 candidates, roughly in order of leverage

1. **Vision (OCR / document understanding).** Unlocks receipt import, PDF assignment reading, and
   statement import — several Phase 1 interfaces (`finance/imports/interfaces.py`) are already
   waiting on this. Start with OCR + a document-structure extractor behind a small provider interface
   (mirroring `core.llm.base.LLMProvider`'s "protocol + swappable implementation" pattern).
2. **Document generation (`work/`, `docs` extra).** DOCX/PDF/PPTX output for reports and assignments,
   using `python-docx`/`pypdf`-style libraries behind their own tool(s), with the same
   execute-then-verify discipline the agent loop already applies (a generated file's existence and
   non-empty size is a trivial, real postcondition).
3. **One real `ComputerAgent` backend** (Fedora first, since that's the primary dev environment) —
   proves out the device-selection design named in `docs/DEVICES.md` before building the others.
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
   #1 and #2 above being solid first.
10. **Additional device backends** (Windows, Phone) once the Fedora backend has validated the
    `ComputerAgent` interface in practice.

## Explicitly deferred, no strong opinion yet

- Voice interface, desktop/mobile UI shells (`interfaces/voice`, `interfaces/desktop`,
  `interfaces/mobile`) — CLI is the only interface until there's a concrete reason to add another.
- Multi-device task routing beyond the capability-matching sketch in `docs/DEVICES.md`.
