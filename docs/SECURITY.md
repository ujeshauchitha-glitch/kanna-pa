# Security

## Permission levels

`core.permissions.levels.PermissionLevel`:

- **LOW** — read/search/list files, create a new file, run allowlisted code, inspect
  screenshots/computer state, move the mouse/scroll/read or write the clipboard/launch an
  application, perform calculations, finance entry/queries, generate a new document. Runs without
  confirmation.
- **REVIEW** — delete files, overwrite an existing file, send messages/emails, upload, submit,
  purchase, click/type/press a key (Kanna can't know the consequence — see `docs/DEVICES.md`), close
  an application (may lose unsaved work), or any other irreversible external action. Requires an
  `ApprovalGate` to explicitly say yes.
- **RESTRICTED** — reserved; nothing in Phase 1 uses it. No policy rule grants it by default (see
  `PermissionPolicy.decide()` — RESTRICTED falls through to `DENY` unless a rule says otherwise).

Every `Tool` declares its `permission` as a class attribute. `PermissionPolicy.decide()` is a small
rule engine: a list of `(predicate, decision)` rules checked in order, falling back to
LOW→ALLOW / REVIEW→REQUIRE_APPROVAL / RESTRICTED→DENY if nothing matches. This is intentionally data,
not a branch of if/else code, so adding a "trusted automation" later — e.g. "always allow
`fs_write_file` with `overwrite=true` under `~/reports/`" — is a matter of appending a `Rule`, not
touching the policy engine. `core/bootstrap.py::default_policy()` currently has exactly one rule,
covering every "creates a file" tool (`fs_write_file`, `document_generate_docx`,
`document_generate_pptx`, `document_generate_pdf`): called with `overwrite=False` (the default,
i.e. creating a new file), each is downgraded from its registered REVIEW default to auto-ALLOW,
because creating a new file is low-risk; an explicit `overwrite=True` on any of them stays gated.

## Approval gates

`core.permissions.gate.ApprovalGate` is a `Protocol`. Four implementations ship:

- `DenyAllGate` — approves nothing. **This is the ultimate fallback** — `bootstrap()` always uses it
  (wrapped, see `TrustStoreGate` below) unless a different fallback gate is explicitly passed in, so
  an unattended/automated run never silently performs a REVIEW-level action it wasn't specifically
  granted.
- `CLIPromptGate` — prompts on stdin/stdout (`kanna --yes ...` wires this in as the fallback).
- `PreApprovedGate` — approves only tool names in an explicit, in-memory allowlist supplied by the
  caller. No persistence, no args matching — the lightweight version for a caller that constructs its
  own registry directly (tests use this).
- `TrustStoreGate` — **trusted automation**, persisted. Consults `TrustRuleRepository` (the
  `trust_rules` table) for a standing approval matching the call's tool name and, optionally, specific
  args; if one matches, the action runs without asking. If none matches, it falls through to another
  gate (`DenyAllGate` or `CLIPromptGate`, per above). `bootstrap()` always wires this in as the
  registry's actual gate — every entry point (an interactive `kanna ask`, a scheduler-triggered run,
  anything else built on `bootstrap()`) shares the same trust store, so a rule granted once applies
  everywhere.

A rule is granted with `kanna trust add <tool_name> [--arg key=value ...] [--note "..."]` — an empty
pattern trusts every call to that tool; one or more `--arg` entries scope the grant to calls where
those specific args match exactly (e.g. `kanna trust add computer_open_application --arg name=firefox`
trusts opening Firefox specifically, not every application). `kanna trust list` shows every granted
rule with its id, scope, note, and grant time; `kanna trust remove <id>` revokes one. Because every row
records *when* and, if given, *why* (`created_at`/`note`) and is never overwritten in place (only
added or deleted), the `trust_rules` table doubles as the audit trail of what's been pre-approved —
not just a runtime allowlist that forgets its own history. See `core/permissions/gate.py`,
`core/permissions/trust.py`, and `core/memory/repositories/trust_rules.py` for the implementation.

**No code path lets a REVIEW-level action run without a gate saying yes.** `ToolRegistry.invoke()`
always calls `PermissionPolicy.decide()` and, if it returns `REQUIRE_APPROVAL`, always calls
`gate.approve()` before executing — there is no bypass. A standing trust rule is still just another
gate decision, made the same way, logged in `execution_log` like any other invocation, and it only
ever *adds* a way for a specific, user-named action to skip asking — it never lowers what fallback
gate would otherwise be used for anything it doesn't cover.

## Filesystem sandboxing

`core.permissions.sandbox.Sandbox` is constructed with a list of allowed root directories (default:
the current working directory + `KANNA_HOME`, see `core/config/settings.py`). Every filesystem tool
resolves the path it's given through `Sandbox.resolve()` before touching disk:

1. Relative paths resolve against the first configured root.
2. The resolved path is fully realized (`Path.resolve(strict=False)`), which follows symlinks.
3. The realized path must be `relative_to()` one of the configured roots, or `SandboxViolation` is
   raised — a symlink inside the sandbox pointing outside it is caught by this, not just literal
   `../` traversal.

`tools/process/run_process.py` never uses a shell (`subprocess.run(..., shell=False)`) and only runs
executables named on an explicit allowlist — there is no metacharacter-injection surface, and no
arbitrary binary can be invoked even if an argument string is attacker-controlled.

## Controlled process execution

Beyond the no-shell/allowlist properties above: every process run has a timeout (default 30s, max
300s, enforced via `subprocess.run(..., timeout=...)`), stdout/stderr are captured and truncated at
200,000 characters rather than allowed to grow unbounded, and the working directory is resolved
through the same `Sandbox`.

## Logging

`core.logging.setup.RedactionFilter` scrubs anything matching an Anthropic API key pattern
(`sk-ant-...`) or a generic `key=`/`token=`/`secret=`/`password=` assignment from every log record
before it's emitted — including to the file handler, so a shared log file doesn't leak credentials.

## What's explicitly out of scope for Phase 1

- No secrets manager / credential vault. `ANTHROPIC_API_KEY` is read from the environment; nothing
  else requires a credential in Phase 1 (finance explicitly never asks for bank login details).
- No multi-user access control — Kanna is a personal, single-user agent.
- No sandboxing of the *process* the interpreters run in beyond the allowlist/timeout above — a
  malicious Python script run via `process_run` has the same OS-level access Kanna itself has. Real
  isolation (containers/VMs) is a reasonable Phase 2+ hardening step if untrusted code is ever a
  requirement, not assumed necessary for a personal agent running the user's own code.
