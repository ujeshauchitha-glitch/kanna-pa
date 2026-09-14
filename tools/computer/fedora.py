"""The first real `ComputerAgent`: X11 desktop automation via `xdotool`/`scrot`/`xclip`.

Named `FedoraAgent` for the device it targets (per `docs/DEVICES.md` —
the user's actual Fedora laptop), but the implementation itself is X11,
not Fedora-specific: it works on any Linux desktop with a live X11
session (Fedora's default GNOME-on-X11, XFCE, or Wayland-with-XWayland
where `xdotool` still functions) and the three tools installed
(`dnf install xdotool scrot xclip` on Fedora). This module was built and
verified against a virtual `Xvfb` display, since the build/CI sandbox
has no real GUI session — see `docs/DEVICES.md` for exactly what was
and wasn't tested where.

Every method checks for a live `DISPLAY` and the specific binary it
needs before running anything, raising `CapabilityUnavailable` with the
concrete reason (no display, binary missing, the underlying command
failed) rather than a bare subprocess traceback or, worse, silently
doing nothing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from core.errors import CapabilityUnavailable
from tools.computer.base import Point

_REQUIRED_BINARIES = ("xdotool", "scrot", "xclip")
_BUTTON_CODES = {"left": "1", "middle": "2", "right": "3"}


def is_available(*, display: str | None = None) -> bool:
    """True if a live X11 session and every required binary are present.

    Used by `core.bootstrap` to decide whether to wire up `FedoraAgent`
    or fall back to `NullComputerAgent` — availability is never assumed,
    only detected.
    """
    if not (display or os.environ.get("DISPLAY")):
        return False
    return all(shutil.which(binary) is not None for binary in _REQUIRED_BINARIES)


class FedoraAgent:
    def __init__(self, *, display: str | None = None, timeout: float = 10.0) -> None:
        self.display = display or os.environ.get("DISPLAY")
        self.timeout = timeout

    # -- plumbing ---------------------------------------------------------

    def _env(self) -> dict:
        env = dict(os.environ)
        if self.display:
            env["DISPLAY"] = self.display
        return env

    def _require(self, binary: str) -> None:
        if shutil.which(binary) is None:
            raise CapabilityUnavailable(
                f"'{binary}' is not installed; computer control needs it (see docs/DEVICES.md)"
            )
        if not self.display:
            raise CapabilityUnavailable(
                "no DISPLAY is set; computer control needs a running X11 session"
            )

    def _run(self, argv: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
        binary = argv[0]
        self._require(binary)
        try:
            return subprocess.run(argv, input=input_bytes, env=self._env(), capture_output=True,
                                   timeout=self.timeout, shell=False)
        except subprocess.TimeoutExpired as exc:
            raise CapabilityUnavailable(f"'{binary}' timed out after {self.timeout}s") from exc
        except OSError as exc:
            raise CapabilityUnavailable(f"could not run '{binary}': {exc}") from exc

    @staticmethod
    def _check(result: subprocess.CompletedProcess, action: str) -> None:
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise CapabilityUnavailable(f"{action} failed: {stderr or f'exit code {result.returncode}'}")

    # -- ComputerAgent protocol --------------------------------------------

    def screenshot(self) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            result = self._run(["scrot", "--overwrite", str(path)])
            self._check(result, "screenshot")
            return path.read_bytes()
        finally:
            path.unlink(missing_ok=True)

    def move_mouse(self, point: Point) -> None:
        result = self._run(["xdotool", "mousemove", str(point.x), str(point.y)])
        self._check(result, "mouse move")

    def click(self, point: Point, button: str = "left") -> None:
        code = _BUTTON_CODES.get(button)
        if code is None:
            raise ValueError(f"unknown mouse button: {button!r} (expected one of {sorted(_BUTTON_CODES)})")
        self.move_mouse(point)
        result = self._run(["xdotool", "click", code])
        self._check(result, "click")

    def type_text(self, text: str) -> None:
        result = self._run(["xdotool", "type", "--", text])
        self._check(result, "type")

    def key_press(self, key: str) -> None:
        result = self._run(["xdotool", "key", key])
        self._check(result, "key press")

    def scroll(self, dx: int, dy: int) -> None:
        # xdotool has no native scroll delta — it's simulated as repeated
        # button-4/5 (vertical) and button-6/7 (horizontal) clicks, the
        # standard X11 convention for a scroll wheel.
        for _ in range(abs(dy)):
            self._check(self._run(["xdotool", "click", "4" if dy < 0 else "5"]), "scroll")
        for _ in range(abs(dx)):
            self._check(self._run(["xdotool", "click", "6" if dx < 0 else "7"]), "scroll")

    def get_clipboard(self) -> str:
        result = self._run(["xclip", "-selection", "clipboard", "-o"])
        self._check(result, "clipboard read")
        return result.stdout.decode("utf-8", errors="replace")

    def set_clipboard(self, text: str) -> None:
        # `xclip` in write mode (no -o) daemonizes to keep serving the
        # selection after this call returns — it double-forks, and the
        # background copy inherits whatever stdout/stderr fds it was
        # given. capture_output=True (used by `_run`) would then hang
        # forever in communicate(), waiting for EOF on a pipe the
        # long-lived daemon still holds open. Redirect to DEVNULL instead
        # so there's nothing to wait on.
        self._require("xclip")
        try:
            subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode("utf-8"),
                            env=self._env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            timeout=self.timeout, shell=False)
        except subprocess.TimeoutExpired as exc:
            raise CapabilityUnavailable(f"'xclip' timed out after {self.timeout}s") from exc
        except OSError as exc:
            raise CapabilityUnavailable(f"could not run 'xclip': {exc}") from exc

    def open_application(self, name: str) -> None:
        binary = shutil.which(name)
        if binary is None:
            raise CapabilityUnavailable(f"'{name}' is not installed or not on PATH")
        if not self.display:
            raise CapabilityUnavailable("no DISPLAY is set; computer control needs a running X11 session")
        try:
            subprocess.Popen([binary], env=self._env(), stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            raise CapabilityUnavailable(f"could not launch '{name}': {exc}") from exc

    def close_application(self, name: str) -> None:
        result = self._run(["xdotool", "search", "--class", name])
        window_ids = result.stdout.decode().split()
        if not window_ids:
            raise CapabilityUnavailable(f"no open window found for application {name!r}")
        for window_id in window_ids:
            self._check(self._run(["xdotool", "windowclose", window_id]), "window close")

    def inspect_screen(self) -> dict:
        geometry = self._run(["xdotool", "getdisplaygeometry"])
        self._check(geometry, "get display geometry")
        width_str, height_str = geometry.stdout.decode().split()

        active = self._run(["xdotool", "getactivewindow", "getwindowname"])
        active_window = active.stdout.decode().strip() if active.returncode == 0 else None

        return {"width": int(width_str), "height": int(height_str), "active_window": active_window}
