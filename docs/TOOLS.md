# Tools

Every tool is registered in `core.tools.registry.ToolRegistry` and reachable exactly one way — through
`registry.invoke(name, args, ctx)`. Run `kanna tools list` for the live catalog (name, permission
level, description) of whatever's registered in your build.

## Filesystem (`tools/filesystem/`) — all sandboxed

| Tool | Permission | What it does |
|---|---|---|
| `fs_read_file` | LOW | Read a UTF-8 text file (truncates past 5MB, reports `truncated`) |
| `fs_write_file` | REVIEW*| Write text to a file |
| `fs_list_directory` | LOW | List a directory's entries, optionally recursive |
| `fs_search_files` | LOW | Regex search across text files under a directory (glob-filterable) |
| `fs_create_directory` | LOW | `mkdir -p` |
| `fs_file_info` | LOW | Existence/type/size/mtime without reading contents |
| `fs_delete` | REVIEW | Delete a file, or a directory tree with `recursive=true` |

\* `fs_write_file`'s registered permission is REVIEW (the safe default), but the default policy
(`core/bootstrap.py::default_policy`) downgrades it — along with the `document_generate_*` tools
below, which follow the identical pattern — to auto-allow when `overwrite` is `False`: creating a
brand-new file is LOW-risk; only overwriting an existing one stays gated. All filesystem tools
resolve paths through `core.permissions.sandbox.Sandbox`, which rejects `..` traversal and symlink
escapes before touching disk.

## Process execution (`tools/process/`)

`process_run` (LOW) runs an allowlisted executable as an argv list — **never through a shell**, so
there's no metacharacter-injection surface. Captures stdout, stderr, exit code, wall-clock duration,
and enforces a timeout (default 30s, max 300s). The default allowlist covers the interpreters/
compilers Phase 1 targets: `python3`, `pytest`, `gcc`/`g++`/`cc`/`c++`, `rustc`/`cargo`, `javac`/
`java`, `node`/`npm`, `octave`/`octave-cli`, plus `echo`/`cat`/`ls`. A caller needing a different set
constructs its own `ProcessTool(allowed_executables=...)` rather than widening the shared default.

## Finance (`finance/tools.py`)

All LOW permission — none of them delete or send anything externally. Each is a thin wrapper over
`finance.service.FinanceService`; see `docs/FINANCE.md` for the deterministic-calculation guarantee
these all rely on.

| Tool | What it does |
|---|---|
| `finance_add_transaction` | Parse and log a spend from natural language |
| `finance_query` | Answer a natural-language spending question |
| `finance_set_budget` | Create a budget for a category (or overall) over a period |
| `finance_add_recurring` | Register a recurring expense |
| `finance_import_csv` | Import transactions from CSV text (deduped) |
| `finance_import_receipt` | Log a transaction by reading a receipt image/PDF (vision-backed, see `docs/VISION.md`) |
| `finance_export` | Export transactions as CSV or JSON |

`finance_import_receipt` additionally needs `ANTHROPIC_API_KEY` (it uses
`vision.document.anthropic_document.AnthropicDocumentProvider` by default) — without one it fails
cleanly with a `VisionUnavailable`-derived error rather than crashing or fabricating a transaction.

## Documents (`documents/tools.py`)

Sandboxed and overwrite-gated exactly like `fs_write_file` (see above) — creating a new file is
auto-allowed, `overwrite=true` requires approval. Fully offline; no network, no LLM, no API key at
render time. See `docs/DOCUMENTS.md`.

| Tool | What it does |
|---|---|
| `document_generate_docx` | Generate a Word document from structured title/sections content |
| `document_generate_pptx` | Generate a PowerPoint deck — one slide per section |
| `document_generate_pdf` | Generate a PDF from the same structured content |

Each raises a clean `document_generation_unavailable` error (never a crash or an empty file) if its
backing library (`python-docx`/`python-pptx`/`reportlab`) isn't installed
(`pip install kanna[documents]`).

## Computer control (`tools/computer/`)

`ComputerAgent` is a Protocol (screenshot, mouse, keyboard, scroll, clipboard, open/close app, inspect
screen) — the device-independent capability surface `FedoraAgent` (real, X11-based) and a future
`WindowsAgent`/`PhoneAgent` implement. Every tool resolves its agent via `get_computer_agent()` at
call time, which picks `FedoraAgent` when a live X11 session and `xdotool`/`scrot`/`xclip` are
detected, `NullComputerAgent` (honest `CapabilityUnavailable`, never a fake success) otherwise. See
`docs/DEVICES.md` for what was actually tested and how.

| Tool | Permission | What it does |
|---|---|---|
| `computer_screenshot` | LOW | Capture a screenshot |
| `computer_move_mouse` | LOW | Move the mouse cursor |
| `computer_click` | REVIEW | Click — Kanna can't know the consequence |
| `computer_type_text` | REVIEW | Type at the current keyboard focus |
| `computer_key_press` | REVIEW | Press a key/combo — could submit a form or trigger a shortcut |
| `computer_scroll` | LOW | Scroll the view |
| `computer_get_clipboard` | LOW | Read the clipboard |
| `computer_set_clipboard` | LOW | Write the clipboard |
| `computer_open_application` | LOW | Launch an application |
| `computer_close_application` | REVIEW | Close an application's windows — may lose unsaved work |
| `computer_inspect_screen` | LOW | Screen dimensions + active window title |

## Building a new tool

1. Implement the `Tool` protocol (`core/tools/protocol.py`): `name`, `description`, `input_schema`/
   `output_schema` (built from `core.tools.schema` helpers), `permission`, `execute(args, ctx)`.
2. Return a `ToolResult` — never raise for an *expected* failure (missing file, invalid input); raise
   only for a genuine bug, which the registry will catch and turn into an `execution_error` result
   anyway, but an intentional `ToolResult.fail(code, message)` gives the caller a stable error code.
3. Register it in `core/bootstrap.py` (or the relevant package's `register_all()`).
4. If it should be reachable without an LLM, add a pattern to `core/planner/rule_based.py`.
5. Write tests: valid input, invalid input, the sandbox/permission boundary if relevant, and the
   failure paths — see `tests/test_fs_tools.py` for the pattern.
