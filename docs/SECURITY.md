# Security

## Permission levels

`core.permissions.levels.PermissionLevel`:

- **LOW** — read/search/list files, create a new file, run allowlisted code, inspect
  screenshots/computer state (once implemented), perform calculations, finance entry/queries. Runs
  without confirmation.
- **REVIEW** — delete files, overwrite an existing file, send messages/emails, upload, submit,
  purchase, or any other irreversible external action. Requires an `ApprovalGate` to explicitly say
  yes.
- **RESTRICTED** — reserved; nothing in Phase 1 uses it. No policy rule grants it by default (see
  `PermissionPolicy.decide()` — RESTRICTED falls through to `DENY` unless a rule says otherwise).

Every `Tool` declares its `permission` as a class attribute. `PermissionPolicy.decide()` is a small
rule engine: a list of `(predicate, decision)` rules checked in order, falling back to
LOW→ALLOW / REVIEW→REQUIRE_APPROVAL / RESTRICTED→DENY if nothing matches. This is intentionally data,
not a branch of if/else code, so adding a "trusted automation" later — e.g. "always allow
`fs_write_file` with `overwrite=true` under `~/reports/`" — is a matter of appending a `Rule`, not
touching the policy engine. `core/bootstrap.py::default_policy()` currently has exactly one rule:
`fs_write_file` with `overwrite=False` (creating a new file) is downgraded from its registered REVIEW
default to auto-ALLOW, because creating a new file is low-risk; overwriting an existing one is not
and stays gated.

## Approval gates

`core.permissions.gate.ApprovalGate` is a `Protocol`. Three implementations ship:

- `DenyAllGate` — approves nothing. **This is the default** for `bootstrap()` unless a gate is
  explicitly passed in, so an unattended/automated run (a scheduled job, a test) never silently
  performs a REVIEW-level action.
- `CLIPromptGate` — prompts on stdin/stdout (`kanna --yes ...` wires this in).
- `PreApprovedGate` — approves only tool names in an explicit allowlist supplied by the caller (e.g.
  a scheduled workflow that's been configured to trust exactly `finance_add_transaction` and nothing
  else). This is the seam for future "trusted automations" — nothing currently constructs one outside
  tests.

**No code path lets a REVIEW-level action run without a gate saying yes.** `ToolRegistry.invoke()`
always calls `PermissionPolicy.decide()` and, if it returns `REQUIRE_APPROVAL`, always calls
`gate.approve()` before executing — there is no bypass.

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
