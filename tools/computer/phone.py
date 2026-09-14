"""A `ComputerAgent` backend for a connected Android device, via ADB
(Android Debug Bridge).

Every method shells out to the real `adb` client — never a shell, no
metacharacter-injection surface (each argument reaches `adb` as its own
argv element; the one place a full command string is built is the
single string handed to the *device's* remote shell via `adb shell`,
and that string is built with `shlex.quote()` around anything
user-supplied). `is_available()`/`list_devices()` check for the `adb`
binary and at least one device actually in "device" state (not
"unauthorized"/"offline", which `adb devices` reports as real but
unusable states) before anything is attempted.

VALIDATION STATUS — read this before trusting this backend the way the
others in this package are trusted. `FedoraAgent` was validated against
a real Xvfb X server in this project's own build environment (see
docs/DEVICES.md); `WindowsAgent` was validated separately by the
project's author against real Windows. This backend has NOT been
exercised against a real Android device or emulator: the sandbox this
was written in has no `/dev/kvm` (no emulator acceleration available at
all) and its network egress policy explicitly blocks the Android SDK's
own download hosts (`dl.google.com`, `redirector.gvt1.com`), so a live
device is categorically unreachable there — not merely untried for lack
of time. The `adb` commands used below (`input tap/swipe/text/keyevent`,
`exec-out screencap`, `wm size`, `dumpsys window`, `am force-stop`,
`monkey -c LAUNCHER`, `pm list packages`) are standard, stable, and
well-documented across Android 5+, and every command's construction and
output-parsing is covered by tests against a mocked subprocess
(tests/test_phone_agent.py) plus real (if device-less) exercise of the
`adb` client binary itself — but that is not a substitute for running
this against an actual phone or emulator. Treat it as unverified until
someone does.
"""
from __future__ import annotations

import re
import shlex
import shutil
import subprocess

from core.errors import CapabilityUnavailable
from tools.computer.base import Point

_TIMEOUT = 15.0

# input keyevent names: https://developer.android.com/reference/android/view/KeyEvent
_KEY_MAP = {
    "enter": "KEYCODE_ENTER", "return": "KEYCODE_ENTER",
    "backspace": "KEYCODE_DEL", "delete": "KEYCODE_FORWARD_DEL",
    "tab": "KEYCODE_TAB", "space": "KEYCODE_SPACE", "escape": "KEYCODE_ESCAPE",
    "back": "KEYCODE_BACK", "home": "KEYCODE_HOME", "menu": "KEYCODE_MENU",
    "recent_apps": "KEYCODE_APP_SWITCH", "power": "KEYCODE_POWER",
    "volume_up": "KEYCODE_VOLUME_UP", "volume_down": "KEYCODE_VOLUME_DOWN",
    "up": "KEYCODE_DPAD_UP", "down": "KEYCODE_DPAD_DOWN",
    "left": "KEYCODE_DPAD_LEFT", "right": "KEYCODE_DPAD_RIGHT",
}


def _adb_present() -> bool:
    return shutil.which("adb") is not None


def list_devices() -> list[str]:
    """Serial numbers `adb devices` reports as actually usable ("device"
    state) — excludes "unauthorized" (debugging not yet accepted on the
    phone) and "offline", which are real states but not usable ones."""
    if not _adb_present():
        return []
    try:
        result = subprocess.run(["adb", "devices"], capture_output=True,
                                text=True, timeout=_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return []
    devices = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) == 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def is_available() -> bool:
    return bool(list_devices())


class AdbPhoneAgent:
    """Controls one connected Android device — the first `adb devices`
    reports as ready, or a specific `serial` if given (passed as `-s` on
    every invocation; required once more than one device is attached,
    since adb itself refuses to guess which one you mean)."""

    def __init__(self, serial: str | None = None, *, timeout: float = _TIMEOUT) -> None:
        self.serial = serial
        self.timeout = timeout

    # -- plumbing ---------------------------------------------------------

    def _require_adb(self) -> None:
        if not _adb_present():
            raise CapabilityUnavailable("'adb' is not on PATH; Android device control requires it")
        if self.serial is None and not list_devices():
            raise CapabilityUnavailable(
                "no Android device is connected and authorized ('adb devices' lists none in "
                "'device' state — check the cable/network connection, and accept the 'Allow USB "
                "debugging' prompt on the phone if one is waiting)"
            )

    def _adb(self, *args: str) -> subprocess.CompletedProcess:
        self._require_adb()
        cmd = ["adb"]
        if self.serial:
            cmd += ["-s", self.serial]
        cmd += list(args)
        try:
            return subprocess.run(cmd, capture_output=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise CapabilityUnavailable(f"adb command timed out after {self.timeout}s: {exc}") from exc
        except OSError as exc:
            raise CapabilityUnavailable(f"could not run adb: {exc}") from exc

    def _shell(self, command: str) -> subprocess.CompletedProcess:
        # One argv element carrying the full command — adb sends it
        # unchanged to the device's shell, which then does its own
        # parsing (including any embedded pipe). Anything user-supplied
        # going into `command` must already be shlex.quote()'d by the
        # caller, since this device-side shell is a real shell.
        return self._adb("shell", command)

    # -- screen --------------------------------------------------------

    def screenshot(self) -> bytes:
        # exec-out (not `shell`) streams raw stdout without the
        # CRLF-mangling `adb shell` applies to a shell-pipe destination —
        # required to get valid PNG bytes back, not just usually-fine ones.
        result = self._adb("exec-out", "screencap", "-p")
        if not result.stdout.startswith(b"\x89PNG"):
            raise CapabilityUnavailable(
                f"'adb exec-out screencap' did not return a PNG (exit {result.returncode}): "
                f"{result.stderr.decode('utf-8', errors='replace')[:200]}"
            )
        return result.stdout

    def inspect_screen(self) -> dict:
        size_result = self._shell("wm size")
        match = re.search(rb"(\d+)x(\d+)", size_result.stdout)
        if not match:
            raise CapabilityUnavailable(
                f"could not parse 'wm size' output: {size_result.stdout!r}")
        width, height = int(match.group(1)), int(match.group(2))

        active_window = None
        try:
            focus_result = self._shell("dumpsys window")
            focus_match = re.search(rb"mCurrentFocus=.*?\{[^}]*\s([\w.]+)/[\w.$]+\}", focus_result.stdout)
            if focus_match:
                active_window = focus_match.group(1).decode("utf-8", errors="replace")
        except CapabilityUnavailable:
            pass  # width/height are still real and worth returning
        return {"width": width, "height": height, "active_window": active_window}

    # -- pointer --------------------------------------------------------

    def move_mouse(self, point: Point) -> None:
        raise CapabilityUnavailable(
            "a touchscreen has no cursor to move independently of a touch — use click(point) directly"
        )

    def click(self, point: Point, button: str = "left") -> None:
        if button == "left":
            self._shell(f"input tap {int(point.x)} {int(point.y)}")
        elif button == "right":
            # No right-click on a touchscreen; the nearest real
            # equivalent gesture is a long-press at the same point.
            self._shell(f"input swipe {int(point.x)} {int(point.y)} {int(point.x)} {int(point.y)} 500")
        else:
            raise ValueError(f"unknown mouse button: {button!r} (expected 'left' or 'right')")

    def scroll(self, dx: int, dy: int) -> None:
        info = self.inspect_screen()
        cx, cy = info["width"] // 2, info["height"] // 2
        step = 300  # one scroll call = one swipe of a fixed magnitude, like a mouse wheel "notch"
        x2 = cx - (step if dx > 0 else -step if dx < 0 else 0)
        y2 = cy - (step if dy > 0 else -step if dy < 0 else 0)
        if (x2, y2) == (cx, cy):
            return
        self._shell(f"input swipe {cx} {cy} {x2} {y2} 200")

    # -- keyboard --------------------------------------------------------

    def type_text(self, text: str) -> None:
        self._shell(f"input text {shlex.quote(text)}")

    def key_press(self, key: str) -> None:
        mapped = _KEY_MAP.get(key.lower())
        if mapped is not None:
            code = mapped
        elif key.upper().startswith("KEYCODE_"):
            code = key.upper()
        else:
            code = f"KEYCODE_{key.upper()}"
        self._shell(f"input keyevent {shlex.quote(code)}")

    # -- clipboard --------------------------------------------------------

    def get_clipboard(self) -> str:
        raise CapabilityUnavailable(
            "Android clipboard access has no reliable API reachable over adb alone on a stock, "
            "non-rooted device"
        )

    def set_clipboard(self, text: str) -> None:
        raise CapabilityUnavailable(
            "Android clipboard access has no reliable API reachable over adb alone on a stock, "
            "non-rooted device"
        )

    # -- apps --------------------------------------------------------

    def _resolve_package(self, name: str) -> str:
        if "." in name and " " not in name:
            return name  # already looks like a package id, e.g. com.android.chrome
        result = self._shell(f"pm list packages {shlex.quote(name)}")
        matches = [line.removeprefix("package:").strip()
                   for line in result.stdout.decode("utf-8", errors="replace").splitlines()
                   if line.startswith("package:")]
        if not matches:
            raise CapabilityUnavailable(f"no installed package matches {name!r}")
        return matches[0]

    def open_application(self, name: str) -> None:
        package = self._resolve_package(name)
        result = self._shell(
            f"monkey -p {shlex.quote(package)} -c android.intent.category.LAUNCHER 1")
        if result.returncode != 0 or b"No activities found" in result.stdout:
            raise CapabilityUnavailable(
                f"could not launch {package!r}: "
                f"{result.stdout.decode('utf-8', errors='replace')[:200]}"
            )

    def close_application(self, name: str) -> None:
        package = self._resolve_package(name)
        self._shell(f"am force-stop {shlex.quote(package)}")


def get_phone_agent(serial: str | None = None) -> AdbPhoneAgent:
    """Construct the agent for the (optionally specific) connected
    device. Like WindowsAgent/FedoraAgent, there's no separate "null"
    fallback here — every method already checks device availability
    itself via `_require_adb()` and reports `CapabilityUnavailable`
    honestly, so an agent handed back when nothing is connected simply
    fails cleanly the moment something is asked of it."""
    return AdbPhoneAgent(serial=serial)
