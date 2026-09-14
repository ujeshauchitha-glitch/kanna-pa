"""Optional per-user launch-at-login registration.

Two real backends, both the standard *per-user* mechanism for their
platform — no administrator/root privileges required, and nothing here
is ever called except from an explicit user action (a button click in
`startup_dialog.py`); enabling is never automatic or silent:

- Windows: a value under `HKCU\\...\\CurrentVersion\\Run` (via the
  stdlib `winreg` — no new dependency).
- Linux: an XDG autostart `.desktop` file under `~/.config/autostart/`
  (freedesktop.org Desktop Entry spec — every major desktop session
  honors this uniformly).

macOS is not implemented; `NullStartupBackend` says so honestly rather
than silently doing nothing or claiming success.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Protocol

_APP_NAME = "Kanna"


class StartupBackend(Protocol):
    def is_supported(self) -> bool: ...
    def unsupported_reason(self) -> str: ...
    def mechanism_description(self) -> str: ...
    def is_enabled(self) -> bool: ...
    def enable(self) -> str | None: ...
    def disable(self) -> str | None: ...


def _launch_command_parts() -> list[str] | None:
    """[python, main.py, "app"] for relaunching this exact install, or
    None if main.py can't be located (an unusual packaging layout) —
    callers must fail cleanly rather than register a broken command."""
    main_py = Path(__file__).resolve().parents[2] / "main.py"
    if not main_py.is_file():
        return None
    python = sys.executable
    if sys.platform == "win32":
        # Avoid a flashing console window at login — pythonw.exe is the
        # windowless twin that ships alongside python.exe in the same
        # install, when present.
        pythonw = Path(python).with_name("pythonw.exe")
        if pythonw.is_file():
            python = str(pythonw)
    return [python, str(main_py), "app"]


def _quote_windows(parts: list[str]) -> str:
    return " ".join(f'"{p}"' if (" " in p or "\t" in p) else p for p in parts)


def _quote_desktop_exec(parts: list[str]) -> str:
    """Desktop Entry Exec key quoting (freedesktop.org spec): wrap an
    argument containing a reserved character in double quotes,
    backslash-escaping any literal backslash/backtick/dollar/quote
    inside it first."""
    reserved = set(' \t\n"\'\\$`><~|&;*?#()[]')

    def quote(part: str) -> str:
        if not any(c in reserved for c in part):
            return part
        escaped = part.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
        return f'"{escaped}"'

    return " ".join(quote(p) for p in parts)


class NullStartupBackend:
    def __init__(self, reason: str):
        self._reason = reason

    def is_supported(self) -> bool:
        return False

    def unsupported_reason(self) -> str:
        return self._reason

    def mechanism_description(self) -> str:
        return "no mechanism available"

    def is_enabled(self) -> bool:
        return False

    def enable(self) -> str | None:
        return self._reason

    def disable(self) -> str | None:
        return None  # nothing was ever registered — trivially "disabled"


class WindowsStartupBackend:
    _RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"

    def is_supported(self) -> bool:
        return sys.platform == "win32"

    def unsupported_reason(self) -> str:
        return "This backend only applies on Windows."

    def mechanism_description(self) -> str:
        return rf"the per-user Windows Startup entry (HKEY_CURRENT_USER\{self._RUN_KEY}), no administrator rights needed"

    def is_enabled(self) -> bool:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self._RUN_KEY) as key:
                winreg.QueryValueEx(key, _APP_NAME)
            return True
        except FileNotFoundError:
            return False

    def enable(self) -> str | None:
        parts = _launch_command_parts()
        if parts is None:
            return "could not locate main.py in this install to launch at startup"
        import winreg
        try:
            with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, self._RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
                winreg.SetValueEx(key, _APP_NAME, 0, winreg.REG_SZ, _quote_windows(parts))
        except OSError as exc:
            return f"could not write the startup registry entry: {exc}"
        return None

    def disable(self) -> str | None:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, self._RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
                winreg.DeleteValue(key, _APP_NAME)
        except FileNotFoundError:
            pass  # already not registered — disabling is idempotent
        except OSError as exc:
            return f"could not remove the startup registry entry: {exc}"
        return None


class LinuxAutostartBackend:
    def is_supported(self) -> bool:
        return sys.platform.startswith("linux")

    def unsupported_reason(self) -> str:
        return "This backend only applies on Linux desktop sessions."

    def mechanism_description(self) -> str:
        return "an XDG autostart entry (~/.config/autostart/kanna.desktop), no root privileges needed"

    def _autostart_dir(self) -> Path:
        xdg_config = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        return Path(xdg_config) / "autostart"

    def _autostart_file(self) -> Path:
        return self._autostart_dir() / "kanna.desktop"

    def is_enabled(self) -> bool:
        return self._autostart_file().is_file()

    def enable(self) -> str | None:
        parts = _launch_command_parts()
        if parts is None:
            return "could not locate main.py in this install to launch at startup"
        content = (
            "[Desktop Entry]\n"
            "Type=Application\n"
            f"Name={_APP_NAME}\n"
            f"Exec={_quote_desktop_exec(parts)}\n"
            "X-GNOME-Autostart-enabled=true\n"
            "Comment=Launch the Kanna task workspace at login\n"
        )
        try:
            self._autostart_dir().mkdir(parents=True, exist_ok=True)
            self._autostart_file().write_text(content)
        except OSError as exc:
            return f"could not write the autostart file: {exc}"
        return None

    def disable(self) -> str | None:
        try:
            self._autostart_file().unlink(missing_ok=True)
        except OSError as exc:
            return f"could not remove the autostart file: {exc}"
        return None


def create_startup_backend() -> StartupBackend:
    if sys.platform == "win32":
        return WindowsStartupBackend()
    if sys.platform.startswith("linux"):
        return LinuxAutostartBackend()
    return NullStartupBackend(
        "Launch at login is not implemented on this platform yet; launch Kanna directly instead.")
