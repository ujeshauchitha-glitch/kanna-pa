# Devices

## Status: three real backends — `FedoraAgent` (X11), `WindowsAgent` (PowerShell), `AdbPhoneAgent` (Android, **unverified** — see below)

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

Three implementations exist:

- **`tools/computer/fedora.py::FedoraAgent`** — a real backend, built on `xdotool` (mouse/keyboard/
  window management), `scrot` (screenshots), and `xclip` (clipboard). Named for the device it targets
  (the user's actual Fedora laptop, per the original spec), but the implementation is X11-generic —
  it works on any Linux desktop with a live X11 session and those three tools installed
  (`dnf install xdotool scrot xclip` on Fedora; `apt install xdotool scrot xclip` on Debian/Ubuntu).
  Every method checks for a live `DISPLAY` and its specific binary before running anything, raising
  `CapabilityUnavailable` with the concrete reason otherwise.
- **`tools/computer/windows.py::WindowsAgent`** — a real backend, built on PowerShell/.NET. Uses
  `System.Drawing` for screenshots, `System.Windows.Forms.Cursor`/`SendKeys` for mouse/keyboard,
  `Get-Clipboard`/`Set-Clipboard` for clipboard, and `Start-Process`/`Stop-Process` for app
  management. Requires only `powershell.exe` on PATH (standard on Windows 10+). Every method checks
  for PowerShell before running anything, raising `CapabilityUnavailable` otherwise.
- **`tools/computer/null.py::NullComputerAgent`** — the honest fallback when no display/tooling is
  detected. Every method raises `CapabilityUnavailable`.

`tools/computer/get_computer_agent()` is the one place that picks between them — it checks
`fedora_available()` first (X11 + all three binaries), then `windows_available()` (PowerShell on
PATH), and falls back to `NullComputerAgent`. Every `computer_*` tool resolves its agent through
this at call time, so a plan built before a display existed (or one built where it didn't) still
does the right thing.

A fourth implementation, `tools/computer/phone.py::AdbPhoneAgent`, also exists but is deliberately
**not** wired into `get_computer_agent()` — see the next section for why and for its tools.

## Phone: `AdbPhoneAgent` (Android, via ADB) — a separate `phone_*` namespace

`tools/computer/phone.py::AdbPhoneAgent` implements the same `ComputerAgent` protocol against a
connected Android device, entirely through the real `adb` client (`input tap/swipe/text/keyevent`,
`exec-out screencap -p`, `wm size`, `dumpsys window`, `pm list packages`, `monkey -c LAUNCHER`,
`am force-stop`). `is_available()`/`list_devices()` require both the `adb` binary on PATH and at
least one device reported in `adb devices`' "device" state (not "unauthorized"/"offline", which are
real-but-unusable states) before anything is attempted.

Two methods can never succeed on this backend and raise `CapabilityUnavailable` unconditionally
rather than pretending: `move_mouse()` (a touchscreen has no cursor independent of a touch) and
`get_clipboard()`/`set_clipboard()` (no reliable API reachable over adb alone on a stock,
non-rooted device). `click()` maps `button="right"` to a long-press swipe at the same point, the
closest real touchscreen equivalent.

**It is deliberately kept out of `get_computer_agent()`'s Fedora → Windows → Null selection chain.**
That function answers "what controls the machine Kanna itself runs on" — a single-device question. A
phone is a categorically different, *additional* device, not another candidate for the same slot,
and folding it in would require guessing which device a given instruction actually means (the
capability-based router this project's docs already flag as unbuilt future work, below). So phone
control is exposed through its own tool namespace instead — `tools/computer/phone_tools.py`,
registered in `core/bootstrap.py` right after `computer_tools` — mirroring the precedent that
`browser_*` tools already live separately from `computer_*` tools. `phone_tap` takes `long_press:
bool` rather than desktop's `button="left"|"right"|"middle"`, since that's the gesture an app
developer (or a phone user) actually recognizes, and `phone_move_mouse`/phone clipboard tools are
simply never registered, rather than registering a tool that can structurally never work.

### Validation status — read this before trusting it

Every other backend in this file was validated against something real: `FedoraAgent` against a live
Xvfb X server in this project's own environment (below), `WindowsAgent` separately by the project's
author against real Windows. **`AdbPhoneAgent` has not been exercised against a real Android device
or emulator.** The sandbox it was built in has no `/dev/kvm` (no emulator acceleration available at
all — a software-rendered emulator was also considered and ruled out, not just slow) and its network
egress policy explicitly blocks the Android SDK's own download hosts (`dl.google.com`,
`redirector.gvt1.com` both return HTTP 403 "organization policy" through the agent proxy), so a live
device is categorically unreachable there, not merely untried for lack of time.

What *was* verified for real in that environment:

- The `adb` client binary itself is genuinely installed (`apt install android-tools-adb`, from
  Ubuntu's own repos — separate from Google's blocked download hosts) and runs for real:
  `list_devices()`/`is_available()` are tested against it unmocked in
  `tests/test_phone_agent.py`, and correctly report zero devices / not-available with nothing
  connected — no mocking involved for that one honest fact.
- Every other method's command construction and output-parsing (the `wm size` / `dumpsys window
  mCurrentFocus` regexes, tap/swipe coordinate math, `shlex.quote()`-based text quoting, key-name
  mapping, package resolution) is covered by tests against a mocked `subprocess.run`
  (`tests/test_phone_agent.py`, ~30 tests) — this confirms the code sends the commands the Android
  documentation says are correct, but a mock cannot confirm a real device does what the docs say.
- End-to-end through the actual tool registry: all eight `phone_*` tools resolve, and with no device
  attached, `phone_screenshot`/`phone_inspect_screen` etc. return a clean `ToolResult.fail` with a
  `CapabilityUnavailable` reason — the same "never fake success" path proven for every other backend.

The `adb` commands used are standard, stable, and well-documented across Android 5+, so this is a
reasonable-confidence implementation — but it is unverified against real hardware until someone with
a device or a working emulator runs it. Treat it that way.

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
| `phone_screenshot`, `phone_inspect_screen`, `phone_scroll`, `phone_open_app` | LOW | Read-only, or trivially reversible (scrolling a view, launching an app you can just close again) |
| `phone_tap`, `phone_type_text`, `phone_key_press` | REVIEW | Same reasoning as `computer_click`/`type_text`/`key_press` — Kanna can't know what a tap or keystroke does |
| `phone_close_app` | REVIEW | May discard unsaved work |

## What this means today

Computer-control tools are registered in the tool registry (`core/bootstrap.py`) and reachable via
`kanna computer <screenshot|inspect|click|move|type|key|open|close|clipboard>` directly, or through
the agent loop's normal plan-step path (where the REVIEW-level ones require an `ApprovalGate` to say
yes, same as everything else). On a machine with no display, every call fails cleanly with
`CapabilityUnavailable` — never a silent no-op reported as success.

## Planned next (device selection, other backends)

The same high-level Kanna request should eventually work regardless of which device executes it —
"take a picture of this receipt" selects a phone, "open this Windows application" selects a Windows
backend, and so on. The capability-based router across multiple registered devices doesn't exist yet;
today `get_computer_agent()` checks Fedora → Windows → Null in order, and phone control is a
separate `phone_*` namespace the planner must call explicitly rather than a candidate in that chain.
Building the full multi-device router — and getting `AdbPhoneAgent` verified against real hardware —
is future work; see `docs/ROADMAP.md`.
