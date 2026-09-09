# Devices

## Status: interface only

Phase 1 defines the device-independent computer-control surface and ships nothing that pretends to
use it. `tools/computer/base.py` declares:

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

`tools/computer/null.py` implements it as `NullComputerAgent`, which raises
`CapabilityUnavailable` from every method with a clear message. It is intentionally **not** wired
into the tool registry — there's nothing useful for it to do until a real backend exists, and Kanna
does not ship a fake one. This matters for the higher-level promise the spec makes: Kanna should
never claim a click or a screenshot happened when it didn't.

## Planned shape (Phase 2+)

```
ComputerAgent (Protocol)
  ├── FedoraAgent    (Linux — likely via a combination of `xdotool`/`ydotool`, X11/Wayland
  │                    screenshot APIs, and clipboard tools depending on the display server)
  ├── WindowsAgent    (Windows — likely via `pywinauto`/`pyautogui`-style automation, or the
  │                    Windows UI Automation API for a more robust element-based approach)
  └── PhoneAgent      (Android/iOS — scope TBD; likely a companion app exposing a small local API
                        for "take a photo of X", "open app Y", rather than full remote control)
```

The same high-level Kanna request should work regardless of which device executes it. Device
selection is not yet designed in code; the intended shape is a capability-based router — e.g. "run my
Octave assignment" selects a device that has `octave` on its `PATH` (discoverable the same way
`tools/process/run_process.py` already checks executable availability), "open this Windows
application" selects a `WindowsAgent`, "take a picture of this receipt" selects a `PhoneAgent`. This
router does not exist yet; it's a Phase 2+ addition to `core/planner` or a new `core/devices` module,
once there is more than one real backend to route between.

## What this means today

Any request that would require computer control, browser automation reliant on a GUI, or a phone
camera fails fast with `CapabilityUnavailable` surfaced through the normal `ToolResult.fail()` /
agent-loop `BLOCKED`/`FAILED` path — never a silent no-op reported as success.
