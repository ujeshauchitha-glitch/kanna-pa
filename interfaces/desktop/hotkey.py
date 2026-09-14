"""Cross-platform global hotkey registration for the desktop app.

Each backend implements the same tiny surface — `register()` (returns
`None` on success or a human-readable reason on failure) and `poll()`
(call periodically from the Tk main loop; invokes the callback if the
hotkey fired since the last poll) — so `app.py` never branches on
platform itself. `create_hotkey_backend()` picks the real backend for
the current platform/session and is honest when none applies: a Wayland
session (even one running XWayland — an X11 key grab only intercepts
input inside XWayland-rendered windows, not session-wide, so it would be
a lie to claim it works there) or an unimplemented platform (macOS) gets
a backend that reports *why* it's unavailable, once, rather than
silently doing nothing forever.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Protocol

DEFAULT_HOTKEY = "ctrl+shift+k"
_ENV_VAR = "KANNA_DESKTOP_HOTKEY"

_MODIFIER_ALIASES = {
    "ctrl": "control", "control": "control",
    "shift": "shift",
    "alt": "alt", "option": "alt",
    "super": "super", "win": "super", "cmd": "super", "meta": "super",
}


def configured_hotkey() -> str:
    return os.environ.get(_ENV_VAR, DEFAULT_HOTKEY)


def parse_hotkey(spec: str) -> tuple[frozenset[str], str]:
    """"ctrl+shift+k" -> ({"control", "shift"}, "k"). Raises ValueError
    for anything that isn't at least one modifier plus one key."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"{spec!r} needs at least one modifier and a key, e.g. 'ctrl+shift+k'")
    *mod_parts, key = parts
    try:
        mods = frozenset(_MODIFIER_ALIASES[m] for m in mod_parts)
    except KeyError as exc:
        raise ValueError(f"unknown modifier {exc.args[0]!r} in {spec!r}") from exc
    if len(key) != 1 or not key.isalnum():
        raise ValueError(f"{spec!r} must end in a single letter or digit, got {key!r}")
    return mods, key


class HotkeyBackend(Protocol):
    def register(self) -> str | None: ...
    def poll(self) -> None: ...
    def unregister(self) -> None: ...


class NullHotkeyBackend:
    """Always unavailable, with an explanation. Used for Wayland,
    unimplemented platforms, and as the fallback when the configured
    hotkey spec itself doesn't parse."""

    def __init__(self, reason: str):
        self.reason = reason

    def register(self) -> str | None:
        return self.reason

    def poll(self) -> None:
        pass

    def unregister(self) -> None:
        pass


class WindowsHotkeyBackend:
    """RegisterHotKey/PeekMessageW via ctypes — no extra dependency."""

    _MOD_BITS = {"alt": 0x0001, "control": 0x0002, "shift": 0x0004, "super": 0x0008}
    _HOTKEY_ID = 1

    def __init__(self, mods: frozenset[str], key: str, on_trigger: Callable[[], None]):
        self._mods = mods
        self._key = key
        self._on_trigger = on_trigger
        self._registered = False

    def register(self) -> str | None:
        import ctypes
        mod_flags = 0
        for m in self._mods:
            mod_flags |= self._MOD_BITS[m]
        vk = ord(self._key.upper())
        ok = bool(ctypes.windll.user32.RegisterHotKey(None, self._HOTKEY_ID, mod_flags, vk))
        if not ok:
            code = ctypes.windll.kernel32.GetLastError()
            return (f"the hotkey is already registered by another application "
                    f"(Windows error {code}); close it or set {_ENV_VAR} to a different combination.")
        self._registered = True
        return None

    def poll(self) -> None:
        if not self._registered:
            return
        import ctypes
        from ctypes import wintypes
        msg = wintypes.MSG()
        # WM_HOTKEY = 0x0312; PM_REMOVE = 1. Non-blocking: returns
        # immediately whether or not a message was waiting.
        if ctypes.windll.user32.PeekMessageW(ctypes.byref(msg), None, 0x0312, 0x0312, 1):
            self._on_trigger()

    def unregister(self) -> None:
        if self._registered:
            import ctypes
            ctypes.windll.user32.UnregisterHotKey(None, self._HOTKEY_ID)
            self._registered = False


class X11HotkeyBackend:
    """XGrabKey on the root window via the optional `python-xlib`
    package. X11 sessions only — see `create_hotkey_backend`."""

    _MOD_ATTRS = {"control": "ControlMask", "shift": "ShiftMask",
                  "alt": "Mod1Mask", "super": "Mod4Mask"}

    def __init__(self, mods: frozenset[str], key: str, on_trigger: Callable[[], None]):
        self._mods = mods
        self._key = key
        self._on_trigger = on_trigger
        self._display = None
        self._root = None
        self._keycode = None
        self._grabbed_masks: list[int] = []
        self._X = None

    def register(self) -> str | None:
        try:
            from Xlib import X, XK
            from Xlib import display as xdisplay
        except ImportError:
            return "python-xlib is not installed; run `pip install kanna[desktop]`."
        try:
            disp = xdisplay.Display()
        except Exception as exc:
            return f"could not open the X display: {exc}"

        keysym = XK.string_to_keysym(self._key)
        if keysym == 0:
            disp.close()
            return f"{self._key!r} is not a recognized X11 key name."
        keycode = disp.keysym_to_keycode(keysym)
        base_mask = 0
        for m in self._mods:
            base_mask |= getattr(X, self._MOD_ATTRS[m])
        root = disp.screen().root
        # Num Lock (Mod2) and Caps Lock (Lock) are independent toggles
        # X11 folds into the modifier state; grab every combination so
        # the hotkey still fires regardless of their state.
        combos = [base_mask, base_mask | X.LockMask, base_mask | X.Mod2Mask,
                  base_mask | X.LockMask | X.Mod2Mask]
        try:
            for mask in combos:
                root.grab_key(keycode, mask, True, X.GrabModeAsync, X.GrabModeAsync)
            disp.sync()
        except Exception as exc:
            disp.close()
            return f"the hotkey is already grabbed by another application: {exc}"

        self._display, self._root, self._keycode = disp, root, keycode
        self._grabbed_masks, self._X = combos, X
        while disp.pending_events():
            disp.next_event()  # drain the grab's own MappingNotify
        return None

    def poll(self) -> None:
        if self._display is None:
            return
        self._display.flush()
        while self._display.pending_events():
            event = self._display.next_event()
            if event.type == self._X.KeyPress and event.detail == self._keycode:
                self._on_trigger()

    def unregister(self) -> None:
        if self._display is None:
            return
        try:
            for mask in self._grabbed_masks:
                self._root.ungrab_key(self._keycode, mask)
            self._display.sync()
        except Exception:
            pass
        finally:
            self._display.close()
            self._display = None


def create_hotkey_backend(on_trigger: Callable[[], None], *, spec: str | None = None) -> HotkeyBackend:
    spec = spec if spec is not None else configured_hotkey()
    try:
        mods, key = parse_hotkey(spec)
    except ValueError as exc:
        return NullHotkeyBackend(f"{_ENV_VAR}={spec!r} is invalid: {exc}")

    if sys.platform == "win32":
        return WindowsHotkeyBackend(mods, key, on_trigger)
    if sys.platform == "darwin":
        return NullHotkeyBackend(
            "Global hotkey is not implemented on macOS yet; launch Kanna directly instead.")
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session == "wayland":
        return NullHotkeyBackend(
            "Global hotkey needs an X11 session; this is Wayland, where an X11 key grab only "
            "works inside XWayland windows, not session-wide. Launch Kanna directly instead.")
    return X11HotkeyBackend(mods, key, on_trigger)
