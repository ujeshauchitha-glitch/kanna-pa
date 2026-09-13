# Desktop task workspace

Run `python main.py app` using a Python installation with Tcl/Tk. The native UI requires no new
packages; the optional `[desktop]` extra provides the tray icon, and `[voice]` provides microphone
recording. Configure `[llm]` and a provider for open-ended requests and authoring. Basic file and
finance commands remain available when the planner falls back to rules.

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
submit/result and transcript handling, and checks minimum-window result space; it skips explicitly
if Tcl/Tk or a display cannot start. `test_provider_routing.py` tests source-authoring provider
selection and local-to-cloud fallback separation using scripted providers, without paid API calls.
