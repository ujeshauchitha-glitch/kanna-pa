"""Windows `ComputerAgent` backend: desktop automation via PowerShell/.NET.

Uses native Windows capabilities through `powershell.exe`:
- Screenshots via `System.Drawing.Graphics.CopyFromScreen`
- Mouse/keyboard via `System.Windows.Forms.Cursor` / `SendKeys`
- Clipboard via `Get-Clipboard` / `Set-Clipboard`
- Screen info via `System.Windows.Forms.Screen`
- App launch/close via `Start-Process` / `Get-Process` / `Stop-Process`

Every method checks for `powershell.exe` on PATH before running anything,
raising `CapabilityUnavailable` with the concrete reason otherwise.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from core.errors import CapabilityUnavailable
from tools.computer.base import Point

_POWERSHELL = "powershell.exe"
_TIMEOUT = 10.0


def is_available() -> bool:
    """True if `powershell.exe` is on PATH (i.e. we're on Windows)."""
    return shutil.which(_POWERSHELL) is not None


class WindowsAgent:
    def __init__(self, *, timeout: float = _TIMEOUT) -> None:
        self.timeout = timeout

    # -- plumbing ---------------------------------------------------------

    def _require_powershell(self) -> None:
        if shutil.which(_POWERSHELL) is None:
            raise CapabilityUnavailable(
                "'powershell.exe' is not on PATH; Windows computer control requires PowerShell"
            )

    def _run_ps(self, script: str, *, check: bool = True) -> subprocess.CompletedProcess:
        """Run a PowerShell script and return the result."""
        self._require_powershell()
        try:
            result = subprocess.run(
                [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, timeout=self.timeout, shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CapabilityUnavailable(f"PowerShell timed out after {self.timeout}s") from exc
        except OSError as exc:
            raise CapabilityUnavailable(f"could not run PowerShell: {exc}") from exc
        if check and result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise CapabilityUnavailable(
                f"PowerShell command failed: {stderr or f'exit code {result.returncode}'}"
            )
        return result

    def _run_ps_unchecked(self, script: str) -> subprocess.CompletedProcess:
        return self._run_ps(script, check=False)

    # -- ComputerAgent protocol --------------------------------------------

    def screenshot(self) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            ps = (
                f"Add-Type -AssemblyName System.Windows.Forms; "
                f"Add-Type -AssemblyName System.Drawing; "
                f"$screen = [System.Windows.Forms.Screen]::PrimaryScreen; "
                f"$bitmap = New-Object System.Drawing.Bitmap($screen.Bounds.Width, $screen.Bounds.Height); "
                f"$graphics = [System.Drawing.Graphics]::FromImage($bitmap); "
                f"$graphics.CopyFromScreen($screen.Bounds.Location, "
                f"[System.Drawing.Point]::Empty, $screen.Bounds.Size); "
                f"$bitmap.Save('{path}'); "
                f"$graphics.Dispose(); $bitmap.Dispose()"
            )
            self._run_ps(ps)
            data = path.read_bytes()
            if not data[:4] == b"\x89PNG":
                raise CapabilityUnavailable("screenshot did not produce a valid PNG")
            return data
        finally:
            path.unlink(missing_ok=True)

    def move_mouse(self, point: Point) -> None:
        ps = (
            f"Add-Type -AssemblyName System.Windows.Forms; "
            f"[System.Windows.Forms.Cursor]::Position = "
            f"New-Object System.Drawing.Point({point.x}, {point.y})"
        )
        self._run_ps(ps)

    def click(self, point: Point, button: str = "left") -> None:
        if button not in ("left", "middle", "right"):
            raise ValueError(f"unknown mouse button: {button!r} (expected one of ['left', 'middle', 'right'])")
        self.move_mouse(point)
        # Use mouse_event via user32.dll for the actual click
        button_code = {"left": "0x02", "middle": "0x20", "right": "0x08"}[button]
        button_up = {"left": "0x04", "middle": "0x40", "right": "0x10"}[button]
        ps = (
            f"Add-Type @'using System.Runtime.InteropServices; "
            f"public class User32 {{ "
            f"[DllImport(\"user32.dll\")] public static extern void mouse_event(uint dwFlags, int dx, int dy, uint dwData, int dwExtraInfo); "
            f"}}'@ -Language CSharp; "
            f"[User32]::mouse_event({button_code}, 0, 0, 0, 0); "
            f"[User32]::mouse_event({button_up}, 0, 0, 0, 0)"
        )
        self._run_ps(ps)

    def type_text(self, text: str) -> None:
        # Escape text for PowerShell string literal
        escaped = text.replace("'", "''")
        ps = (
            f"Add-Type -AssemblyName System.Windows.Forms; "
            f"[System.Windows.Forms.SendKeys]::SendWait('{escaped}')"
        )
        self._run_ps(ps)

    def key_press(self, key: str) -> None:
        # Map common key names to SendKeys format
        key_map = {
            "enter": "{ENTER}", "return": "{ENTER}",
            "tab": "{TAB}",
            "escape": "{ESC}", "esc": "{ESC}",
            "backspace": "{BS}", "delete": "{DEL}",
            "space": " ",
            "up": "{UP}", "down": "{DOWN}", "left": "{LEFT}", "right": "{RIGHT}",
            "home": "{HOME}", "end": "{END}",
            "pageup": "{PGUP}", "pagedown": "{PGDN}",
            "f1": "{F1}", "f2": "{F2}", "f3": "{F3}", "f4": "{F4}",
            "f5": "{F5}", "f6": "{F6}", "f7": "{F7}", "f8": "{F8}",
            "f9": "{F9}", "f10": "{F10}", "f11": "{F11}", "f12": "{F12}",
            "ctrl": "^", "alt": "%", "shift": "+",
        }
        send_key = key_map.get(key.lower(), key)
        escaped = send_key.replace("'", "''")
        ps = (
            f"Add-Type -AssemblyName System.Windows.Forms; "
            f"[System.Windows.Forms.SendKeys]::SendWait('{escaped}')"
        )
        self._run_ps(ps)

    def scroll(self, dx: int, dy: int) -> None:
        # Use mouse_event with MOUSEWHEEL (0x0800) and MOUSEHWHEEL (0x1000)
        # WHEEL_DELTA = 120
        ps_parts = [
            "Add-Type @'using System.Runtime.InteropServices; "
            "public class User32 { "
            "[DllImport(\"user32.dll\")] public static extern void mouse_event(uint dwFlags, int dx, int dy, uint dwData, int dwExtraInfo); "
            "}'@ -Language CSharp"
        ]
        if dy != 0:
            clicks = max(1, abs(dy))
            wheel_data = 120 if dy > 0 else -120
            ps_parts.append(
                f"for ($i = 0; $i -lt {clicks}; $i++) {{ "
                f"[User32]::mouse_event(0x0800, 0, 0, {wheel_data}, 0) }}"
            )
        if dx != 0:
            clicks = max(1, abs(dx))
            wheel_data = 120 if dx > 0 else -120
            ps_parts.append(
                f"for ($i = 0; $i -lt {clicks}; $i++) {{ "
                f"[User32]::mouse_event(0x1000, 0, 0, {wheel_data}, 0) }}"
            )
        self._run_ps("; ".join(ps_parts))

    def get_clipboard(self) -> str:
        result = self._run_ps_unchecked("Get-Clipboard -Raw")
        if result.returncode != 0:
            # Empty clipboard can cause non-zero exit on some PS versions
            return ""
        return result.stdout.decode("utf-8", errors="replace").strip()

    def set_clipboard(self, text: str) -> None:
        # Escape for PowerShell single-quoted string
        escaped = text.replace("'", "''")
        self._run_ps(f"Set-Clipboard -Value '{escaped}'")

    def open_application(self, name: str) -> None:
        # Try Start-Process; if the name has a path separator, use it directly
        if "/" in name or "\\" in name:
            ps = f"Start-Process -FilePath '{name.replace(chr(39), chr(39)+chr(39))}' -PassThru | Select-Object -ExpandProperty Id"
        else:
            ps = f"Start-Process -FilePath '{name.replace(chr(39), chr(39)+chr(39))}' -PassThru | Select-Object -ExpandProperty Id"
        result = self._run_ps_unchecked(ps)
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise CapabilityUnavailable(f"could not launch '{name}': {stderr}")

    def close_application(self, name: str) -> None:
        # Get processes matching the name, stop them
        escaped = name.replace("'", "''")
        ps = (
            f"$procs = Get-Process -Name '{escaped}' -ErrorAction SilentlyContinue; "
            f"if (-not $procs) {{ Write-Error 'no process found'; exit 1 }}; "
            f"$procs | Stop-Process -Force"
        )
        result = self._run_ps_unchecked(ps)
        if result.returncode != 0:
            raise CapabilityUnavailable(f"no running process found for application '{name}'")

    def inspect_screen(self) -> dict:
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$screen = [System.Windows.Forms.Screen]::PrimaryScreen; "
            "$active = (Get-Process | Where-Object {$_.MainWindowHandle -ne [IntPtr]::Zero} "
            "| Sort-Object MainWindowHandle -Descending | Select-Object -First 1).MainWindowTitle; "
            "Write-Output \"$($screen.Bounds.Width) $($screen.Bounds.Height) $active\""
        )
        result = self._run_ps(ps, check=False)
        output = result.stdout.decode("utf-8", errors="replace").strip()
        parts = output.split(" ", 2)
        try:
            width = int(parts[0])
            height = int(parts[1])
            active_window = parts[2] if len(parts) > 2 else None
            if active_window == "":
                active_window = None
        except (ValueError, IndexError):
            raise CapabilityUnavailable(f"could not parse screen info: {output!r}")
        return {"width": width, "height": height, "active_window": active_window}
