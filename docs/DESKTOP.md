# Desktop task workspace

Run `python main.py app` using a Python installation with Tcl/Tk. The native UI requires no new
packages; the optional `[desktop]` extra provides the tray icon and the X11 global-hotkey backend,
and `[voice]` provides microphone recording. Configure `[llm]` and a provider for open-ended requests
and authoring. Basic file and finance commands remain available when the planner falls back to rules.

## Launching, single instance, and the global hotkey

`main()` (`interfaces/desktop/app.py`) binds a `SingleInstanceGuard` (`interfaces/desktop/singleton.py`)
before constructing anything else — a loopback TCP listener on a port derived deterministically from
the resolved `KANNA_HOME` path. If binding fails, another instance already holds it: the new process
sends it one short message and exits immediately, without ever starting a worker or a window. This is
what actually prevents duplicate instances and duplicate execution — it's not a UI-level check, and it
doesn't depend on the working directory, argv[0], or the on-disk repo location, so `kanna app` behaves
identically whether launched from inside the repository, from an arbitrary folder, or via a shortcut
whose target path contains spaces. A crashed instance releases the OS-level port automatically, so
there's no stale lock file to detect or clean up.

Repeated activation — running `python main.py app` again, pressing the global hotkey, or using the
tray "Show" item — always reaches the *same* window through the same mechanism: an event queued for
the Tk main loop's existing poll (`("show", None)`), never a second window.

The global hotkey (`interfaces/desktop/hotkey.py`, default **Ctrl+Shift+K**, configurable via
`KANNA_DESKTOP_HOTKEY`, e.g. `ctrl+alt+j`) has a real backend on Windows (`RegisterHotKey`/
`PeekMessageW` via `ctypes`, no extra dependency) and on X11 Linux (`XGrabKey` via the optional
`python-xlib` package, part of the `[desktop]` extra) — both validated end-to-end against a live
session (see `tests/test_desktop_hotkey.py`'s skip-if-unavailable live tests). Wayland sessions and
macOS get an explained `NullHotkeyBackend` instead of a silent no-op or a false claim: an X11 key grab
only intercepts input inside XWayland-rendered windows, not session-wide, so it would be dishonest to
register it under Wayland and call it working. Any registration failure — already-grabbed key, no
display, an invalid `KANNA_DESKTOP_HOTKEY` spec — is logged once into the task workspace's own output
panel on startup, not silently swallowed; the app remains fully usable by launching it directly either
way. macOS platform adapters are not implemented; nothing claims otherwise.

## Startup at login

The **Startup** button opens a dialog for the standard *per-user* launch-at-login mechanism —
`interfaces/desktop/startup.py`, never triggered except by clicking Enable/Disable there, never
automatic. Windows: a value under `HKCU\...\CurrentVersion\Run` (via the stdlib `winreg`, no admin
rights, no new dependency). Linux: an XDG autostart `.desktop` file under `~/.config/autostart/`
(no root, honored uniformly by GNOME/KDE/XFCE session startup). Both relaunch the exact running
install (`sys.executable` + this checkout's `main.py app`, using `pythonw.exe` on Windows when
present to avoid a flashing console window) — if `main.py` can't be located (an unusual packaging
layout), Enable fails with that reason rather than registering a broken command. Disable is
idempotent — calling it when nothing is registered is a harmless no-op, not an error. macOS is not
implemented; the dialog says so and disables both buttons rather than pretending.

## Working on a task

Choose a starter or type into the multiline composer. Starters only fill the draft; they do not
execute. Add source files within the configured sandbox folders, then press Run task or Ctrl+Enter.
Paths are supplied to the planner as source references; adding a file does not silently expand the
sandbox or read/upload its contents. Files outside the workspace must first be copied into it.

The connection panel displays the selected planner/model and registered tool count. These indicate
configuration, not verified remote availability. **Connection & access** shows the actual workspace
roots plus an editable LLM provider/model/fallback-models/timeout form
(`interfaces/desktop/connection_dialog.py`) that writes straight to `config.toml`
(`core.config.settings.update_config_file` — a narrow scalar-line writer that preserves every other
line in the file exactly, not a general TOML writer). Saving never takes effect on the
already-running worker (`bootstrap()` only runs once per launch) — the dialog says so explicitly
("Restart Kanna ... to use it") rather than implying a live reload happened.

The progress area shows real agent state events and tool names. The result retains complete/failed/
blocked distinctions and the draft stays available after execution. Copy result copies the last
returned message. Output files are collected only from successful tool outcomes and must still
exist inside the sandbox. Copy path copies a selected file's location; Show folder opens its parent
directory (it does not execute generated code). A partially failed workflow can still expose files
successfully produced by earlier steps.

Record 4 seconds uses the existing Google transcription path, then inserts text into the draft for
review. It never automatically executes recognized speech. Missing dependencies, microphone failures,
and transcription errors leave typed input usable. Recording and task execution cannot overlap.

## Task history

The **Task history** button (top bar) opens a read-only browser over every past run — persisted
`plans`/`plan_steps` rows the agent loop already writes (`core/agent/history.py`), not a new store and
not a rerun. Each run shows its request, final status, every step's tool/status/attempt count and
outcome message, and the files it produced. A file's continued existence is checked live each time
the dialog opens or refreshes; one recorded but since moved or deleted is marked **missing** rather
than silently omitted or claimed present. History survives an app restart (it's the same database the
live session writes to) and is scoped to desktop-submitted runs by default. The dialog queries the
worker's `Database` handle directly from the Tk thread — safe because `Database` is built for that
(its own lock, `check_same_thread=False`) — so browsing history never blocks or is blocked by a task
actually running.

## Approvals and lifecycle

A REVIEW action not already covered by a trust rule opens a dialog showing the tool, exact resolved
arguments, and reason. Allow once approves that invocation; Deny, Escape, or closing the dialog
rejects it. The dialog does not create standing trust. Existing TrustStoreGate rules still apply.

One worker owns startup, session logging, all task executions, and shutdown. Duplicate submissions
are rejected while busy. Worker/tray/voice callbacks send queue messages; only the Tk main thread
touches widgets. Requests and answers are persisted through AgentSession; this is an audit history,
not multi-turn model memory or a restored conversation view.

Window close minimizes to the taskbar so the app remains reachable even without a tray icon. The
optional tray Show action and the global hotkey restore it. Quit waits for the current workflow to
return (it does not cancel it — see the dialog's own wording), denies pending approvals on
shutdown, then closes browser/database resources. It does not promise undo of side effects.

The layout adapts below 720 pixels high with a shorter composer and source/output controls, keeping
the result area visible at the 860×640 minimum. Full layout is 1120×780 by default.

## Cancellation

The **Cancel** button (next to Copy result) is enabled while a task is running. It's cooperative,
not preemptive: `AgentLoop.run()` (`core/agent/loop.py`) checks a `threading.Event` at two specific
points — before the next step starts, and before the next correction retry within a step — never
mid-call. A step already invoking a tool or an LLM always runs to its actual outcome; cancellation
only stops *further* work from starting. The run is then reported `CANCELLED` (a real terminal
`AgentState`, distinct from `FAILED`), and the message says exactly how much completed — e.g.
"Cancelled after 1 of 3 step(s) completed. Actions already taken were not undone." Cancellation
never claims to undo a side effect a tool already performed, because it doesn't attempt to.

A REVIEW approval dialog is application-modal, which makes the main window's Cancel button
unreachable while one is open — so the dialog itself has a **Cancel task** button (denies this one
action *and* requests cancellation of the whole run) alongside Allow once/Deny. A cancellation
requested from anywhere else while an approval is pending is handled automatically: `DesktopGate`
(`interfaces/desktop/worker.py`) denies the pending request and closes its dialog on the next
~0.1s check, the same path already used for shutdown, so a task doesn't hang forever staring at a
question nobody's going to answer.

Because a provider call has no way to be interrupted once sent, cooperative cancellation checked
only *between* calls depends on each call eventually returning at all — so `AnthropicProvider` and
`LiteLLMProvider` now send every request with a real timeout (`llm_timeout_seconds`, default 120s,
`KANNA_LLM_TIMEOUT_SECONDS`/`config.toml`), bounding how long a hung or unreachable model server can
block a step regardless of cancellation. `tools/process/run_process.py` already had its own
independent timeout.

Cancelled runs show up in task history exactly like any other outcome (`⏹` badge) — steps that never
got to run are shown as "pending" with an honest "Not run — the task was cancelled first." message,
not confused with an earlier step having failed.

## Provider integration

Planning, author_content, and assignment_solve now share `core.llm.factory.build_provider` instead
of authoring always selecting Anthropic. LiteLLM's Ollama compatibility URL/key are per-call
arguments. A cloud fallback cannot inherit an earlier local Ollama endpoint or dummy key. No global
LiteLLM settings or configured model names are mutated. PDF/image vision extraction still uses its
existing Anthropic vision provider; this change does not add universal vision routing. Every
provider call now carries `llm_timeout_seconds` (see Cancellation, above) — `build_provider` is the
one place that reads the setting, so both providers stay consistent without each hardcoding it.

## Tests

`test_desktop_worker.py` exercises real agent/database sessions, serial execution, recovery, actual
overwrite approvals and shutdown. `test_desktop_ui.py` constructs real Tk widgets, exercises ready/
submit/result and transcript handling, checks minimum-window result space, and (with
`integrations=True`) exercises the real hotkey-backend wiring end to end; it skips explicitly if
Tcl/Tk or a display cannot start. `test_desktop_singleton.py` exercises the real single-instance
guard's bind/notify/close behavior with real sockets — no mocking, since the OS-level bind conflict
*is* the thing under test. `test_desktop_hotkey.py` covers spec parsing and platform dispatch
deterministically (mocked `sys.platform`/`XDG_SESSION_TYPE`, never a real grab), plus a separate,
explicitly live-only pair of tests that register the real X11 backend against an actual X server and
confirm it fires on the exact configured combo and stays silent on an unrelated key — skipped where
no live X11 session with `python-xlib` and `xdotool` exists. `test_agent_history.py` drives the real
`AgentLoop` against real filesystem tools (no mocking) so `list_runs`/`get_run` are checked against
genuine plan/plan_step rows — including a file deleted between the run and the history read, to prove
the "missing" case is real, not asserted against a canned fixture. `test_desktop_history_dialog.py`
constructs the real dialog against a real in-memory database and checks run selection, the files
list's ✓/✗ markers, refresh-preserves-selection, and copy-path. `test_provider_routing.py` tests source-authoring provider
selection and local-to-cloud fallback separation using scripted providers, without paid API calls,
and that `build_provider` forwards the configured timeout to both providers, and that a real
(mocked-transport) `LiteLLMProvider.complete()` call actually carries it. `test_agent_loop.py`
covers cancellation at the `AgentLoop` level directly (before the next step, before the next
correction retry, cancelled-before-anything-ran, and that omitting `cancel` entirely behaves
exactly as before). `test_desktop_worker.py`/`test_desktop_ui.py` cover it end to end through the
real worker and real dialog — cancelling denies a pending approval without shutting down the
worker, the main-window Cancel button reaches `worker.cancel()`, and the dialog's own Cancel task
button does too. Live-verified by hand: launched the real app, opened a REVIEW approval dialog for
real, clicked Cancel task, and confirmed the target file was never actually written.

`test_desktop_startup.py` exercises both real backends without ever touching a real machine's
actual startup registration: the Windows one against an injected in-memory fake `winreg` (the real
module doesn't exist on non-Windows, and even on Windows a test must never write the user's actual
Startup entry), the Linux one against a `tmp_path` `XDG_CONFIG_HOME`, never `~/.config/autostart`
— enable/disable round-trip, disable-when-never-enabled is a no-op, a missing-`main.py` failure is
reported not silently swallowed, and the `Exec=`/registry-value quoting is checked as a pure
function. `test_desktop_startup_dialog.py` covers the dialog's own state machine (button
enabled/disabled per supported/enabled state, a failed enable shows the real error and does not
claim success) against a small in-memory fake backend. Live-verified by hand: opened the real
dialog in the real app, clicked Enable, confirmed the real `~/.config/autostart/kanna.desktop` file
was written with the correct `Exec=` line, clicked Disable, confirmed it was removed.

`test_settings.py` covers `update_config_file` directly — creating a new file, updating one key
while leaving every other line (including comments) untouched, appending a key not already present,
escaping embedded quotes/backslashes (round-tripped back through real `tomllib`, not just string
comparison), no stray `.tmp` file left behind, and a full round trip through `load_settings()`.
`test_desktop_connection_dialog.py` exercises the real dialog (fields prefilled from real
`Settings`, Save against a real `tmp_path` config file — not a mocked `update_config_file` — an
invalid timeout or empty model rejected with nothing written). Live-verified by hand: opened the
real dialog in the real app, edited the model field, clicked Save, and confirmed the real
`config.toml` was updated correctly with every other key preserved exactly.
