# Devices

## Status: one real backend — `FedoraAgent` (X11)

`tools/computer/base.py` declares the device-independent surface:

```python
class ComputerAgent(Protocol):
    def screenshot(self) -> bytes: ...
    def move_mouse(self, point: Point) -> None: ...
    def click(self, point: Point, button: str = "left") -> None: ...
    def type_text(self, text: str) -> None: ...
    def key_press(self, key: str) -> None: ...
    def scroll(self, dx: int, dy: int) -> None: ...
    def get_clipboard(self) -> str: ...
    def set_clipboard(self, text: str) -> None: ...
    def open_application(self, name: str) -> None: ...
    def close_application(self, name: str) -> None: ...
    def inspect_screen(self) -> dict: ...
```

Two implementations exist:

- **`tools/computer/fedora.py::FedoraAgent`** — a real backend, built on `xdotool` (mouse/keyboard/
  window management), `scrot` (screenshots), and `xclip` (clipboard). Named for the device it targets
  (the user's actual Fedora laptop, per the original spec), but the implementation is X11-generic —
  it works on any Linux desktop with a live X11 session and those three tools installed
  (`dnf install xdotool scrot xclip` on Fedora; `apt install xdotool scrot xclip` on Debian/Ubuntu).
  Every method checks for a live `DISPLAY` and its specific binary before running anything, raising
  `CapabilityUnavailable` with the concrete reason otherwise.
- **`tools/computer/null.py::NullComputerAgent`** — the honest fallback when no display/tooling is
  detected. Every method raises `CapabilityUnavailable`.

`tools/computer/get_computer_agent()` is the one place that picks between them —
`tools/computer/fedora.py::is_available()` checks for a live `DISPLAY` plus all three binaries, with
no assumption either way. Every `computer_*` tool resolves its agent through this at call time, so a
plan built before a display existed (or one built where it didn't) still does the right thing.

## What was actually tested, and how

This build/CI sandbox has no GUI of its own — no `DISPLAY`, no X server, none of `xdotool`/`scrot`/
`xclip` installed by default. Rather than ship `FedoraAgent` untested, a virtual display was set up to
validate it for real:

```bash
apt-get install -y xvfb xdotool scrot xclip fluxbox xterm x11-apps
Xvfb :99 -screen 0 1280x800x24 &
DISPLAY=:99 fluxbox &
```

Against that display, every method was exercised directly and the *effect* was independently
verified, not just "the subprocess call returned 0":

- **`screenshot()`** — returned bytes were checked for a valid PNG header and decoded with Pillow.
- **`move_mouse()` / `scroll()`** — confirmed not to raise (no independently observable side effect
  worth asserting beyond that).
- **`click()` + `type_text()` + `key_press()`** — the strongest test: clicked into a real `xterm`
  running `cat > file`, typed a string, sent Ctrl-D, and read the string back out of the file the
  terminal's own shell wrote. This is the one test that actually proves keyboard focus and X11 event
  delivery work, not just that `xdotool` exited 0.
- **`get_clipboard()` / `set_clipboard()`** — round-tripped a value through the real X clipboard
  selection.
- **`open_application()` / `close_application()`** — launched `xclock`, confirmed its window existed
  via `xdotool search --class`, closed it, confirmed the window was gone.
- **`inspect_screen()`** — confirmed reported dimensions matched the `Xvfb` geometry.
- **The honest-failure paths** — `CapabilityUnavailable` with no `DISPLAY`, and with the required
  binary missing (via `monkeypatch`).

This found and fixed one real bug (see below), and CI now runs the same setup (`xvfb-run` +
`apt-get install xvfb xdotool scrot xclip` in `.github/workflows/tests.yml`) so
`tests/test_fedora_agent.py` runs for real on every push rather than perpetually skipping. Locally,
without a display, that one file's tests skip cleanly (`pytest.mark.skipif`) — the rest of the suite
is unaffected either way.

### The bug this found

`xclip -selection clipboard` (write mode — no `-o`) daemonizes: it forks a background process to keep
serving the clipboard selection after the command that set it exits. That background copy inherits
whatever file descriptors it was launched with — including, originally, the `stdout`/`stderr` pipes
Python's `subprocess.run(..., capture_output=True)` creates. `communicate()` then waits forever for
those pipes to close, which they never do while the daemon holds them open, so `set_clipboard()`
would hang until timeout on every real call. Fixed by redirecting `stdout`/`stderr` to `DEVNULL` for
that one call instead of capturing them — nothing needs xclip's output on a write, and `DEVNULL`
doesn't keep a pipe open for the daemon to inherit. `tests/test_fedora_agent.py::test_clipboard_round_trip`
is what caught this.

## Permission levels for computer control

Following the same "read/reversible is LOW, unpredictable-or-irreversible is REVIEW" split the rest
of Kanna uses (`docs/SECURITY.md`):

| Tool | Level | Why |
|---|---|---|
| `computer_screenshot`, `computer_inspect_screen`, `computer_get_clipboard`, `computer_move_mouse`, `computer_scroll`, `computer_set_clipboard`, `computer_open_application` | LOW | Read-only, or a side effect that's trivially reversible (moving a cursor, scrolling a view, overwriting the clipboard, launching an app you can just close again) |
| `computer_click`, `computer_type_text`, `computer_key_press` | REVIEW | Kanna cannot know what a click, keystroke, or key combo will actually do — it could submit a form, send a message, or trigger anything else the "irreversible external action" category exists for |
| `computer_close_application` | REVIEW | May discard unsaved work |

## What this means today

Computer-control tools are registered in the tool registry (`core/bootstrap.py`) and reachable via
`kanna computer <screenshot|inspect|click|move|type|key|open|close|clipboard>` directly, or through
the agent loop's normal plan-step path (where the REVIEW-level ones require an `ApprovalGate` to say
yes, same as everything else). On a machine with no display, every call fails cleanly with
`CapabilityUnavailable` — never a silent no-op reported as success.

## Planned next (device selection, other backends)

The same high-level Kanna request should eventually work regardless of which device executes it —
"take a picture of this receipt" selects a phone, "open this Windows application" selects a Windows
backend, and so on. That router doesn't exist yet; there is exactly one backend
(`get_computer_agent()` is a fixed choice between `FedoraAgent` and `NullComputerAgent`, not a router
across multiple *registered* devices). Building `WindowsAgent`/`PhoneAgent` and the capability-based
router across them is future work — see `docs/ROADMAP.md`.
