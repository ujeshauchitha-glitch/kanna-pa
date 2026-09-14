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

No startup-at-login integration exists yet (Task Scheduler / systemd user service / launchd are the
natural per-platform mechanisms; see `docs/ROADMAP.md`).

## Working on a task

Choose a starter or type into the multiline composer. Starters only fill the draft; they do not
execute. Add source files within the configured sandbox folders, then press Run task or Ctrl+Enter.
Paths are supplied to the planner as source references; adding a file does not silently expand the
sandbox or read/upload its contents. Files outside the workspace must first be copied into it.

The connection panel displays the selected planner/model and registered tool count. These indicate
configuration, not verified remote availability. Connection & access shows the actual workspace
roots and configuration instructions. Restart after changing provider settings.

The progress area shows real agent state events and tool names. The result retains complete/failed/
blocked distinctions and the draft stays available after execution. Copy result copies the last
returned message. Output files are collected only from successful tool outcomes and must still
exist inside the sandbox. Copy path copies a selected file's location; Show folder opens its parent
directory (it does not execute generated code). A partially failed workflow can still expose files
successfully produced by earlier steps.

Record 4 seconds uses the existing Google transcription path, then inserts text into the draft for
review. It never automatically executes recognized speech. Missing dependencies, microphone failures,
and transcription errors leave typed input usable. Recording and task execution cannot overlap.

## Approvals and lifecycle

A REVIEW action not already covered by a trust rule opens a dialog showing the tool, exact resolved
arguments, and reason. Allow once approves that invocation; Deny, Escape, or closing the dialog
rejects it. The dialog does not create standing trust. Existing TrustStoreGate rules still apply.

One worker owns startup, session logging, all task executions, and shutdown. Duplicate submissions
are rejected while busy. Worker/tray/voice callbacks send queue messages; only the Tk main thread
touches widgets. Requests and answers are persisted through AgentSession; this is an audit history,
not multi-turn model memory or a restored conversation view.

Window close minimizes to the taskbar so the app remains reachable even without a tray icon. The
optional tray Show action and Windows Ctrl+Shift+K restore it. Quit waits for the current workflow
to return, denies pending approvals on shutdown, then closes browser/database resources. It does
not promise cancellation or undo side effects. A hung provider can delay shutdown; interruptible
provider calls and explicit task cancellation remain future work.

The layout adapts below 720 pixels high with a shorter composer and source/output controls, keeping
the result area visible at the 860×640 minimum. Full layout is 1120×780 by default.

## Provider integration

Planning, author_content, and assignment_solve now share `core.llm.factory.build_provider` instead
of authoring always selecting Anthropic. LiteLLM's Ollama compatibility URL/key are per-call
arguments. A cloud fallback cannot inherit an earlier local Ollama endpoint or dummy key. No global
LiteLLM settings or configured model names are mutated. PDF/image vision extraction still uses its
existing Anthropic vision provider; this change does not add universal vision routing.

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
no live X11 session with `python-xlib` and `xdotool` exists. `test_provider_routing.py` tests source-authoring provider
selection and local-to-cloud fallback separation using scripted providers, without paid API calls.
