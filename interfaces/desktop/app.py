"""Kanna desktop app — always-accessible GUI with voice + text input.

Launch with:  python main.py app
Or directly:  python -m interfaces.desktop.app

Features:
- Text input + Send button for typed commands
- Voice button (hold to speak, release to process)
- Live output display with scrolling
- System tray icon (minimize to tray, click to restore)
- Global hotkey: Ctrl+Shift+K to summon/hide the window
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
from pathlib import Path
from typing import Callable

from core.bootstrap import bootstrap, Kanna


class KannaApp:
    def __init__(self) -> None:
        self.kanna: Kanna | None = None
        self._voice_queue: queue.Queue[str] = queue.Queue()
        self._recording = False
        self._hidden = False

        # Build the window
        self.root = tk.Tk()
        self.root.title("Kanna")
        self.root.geometry("700x520")
        self.root.minsize(500, 400)
        self.root.configure(bg="#1e1e2e")

        self._build_ui()
        self._setup_tray()
        self._setup_hotkey()

        # Minimize to tray instead of closing
        self.root.protocol("WM_DELETE_WINDOW", self._minimize_to_tray)

        # Bootstrap Kanna in background
        self._log("Starting Kanna...")
        threading.Thread(target=self._init_kanna, daemon=True).start()

    def _build_ui(self) -> None:
        style = {"bg": "#1e1e2e", "fg": "#cdd6f4", "font": ("Segoe UI", 10)}
        accent = "#89b4fa"
        surface = "#313244"

        # Header
        header = tk.Frame(self.root, bg=surface, height=48)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="Kanna", font=("Segoe UI", 16, "bold"),
                 bg=surface, fg=accent).pack(side=tk.LEFT, padx=12)
        tk.Label(header, text="AI Work Agent", font=("Segoe UI", 10),
                 bg=surface, fg="#6c7086").pack(side=tk.LEFT)

        # Output area
        self.output = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, state=tk.DISABLED,
            bg="#181825", fg="#cdd6f4", insertbackground="#cdd6f4",
            font=("Cascadia Code", 10), relief=tk.FLAT, padx=10, pady=10,
            borderwidth=0, highlightthickness=0,
        )
        self.output.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
        self.output.tag_config("user", foreground="#a6e3a1")
        self.output.tag_config("kanna", foreground="#89b4fa")
        self.output.tag_config("error", foreground="#f38ba8")
        self.output.tag_config("info", foreground="#6c7086")

        # Input area
        input_frame = tk.Frame(self.root, bg="#1e1e2e")
        input_frame.pack(fill=tk.X, padx=8, pady=(4, 8))

        self.entry = tk.Entry(
            input_frame, bg=surface, fg="#cdd6f4", insertbackground="#cdd6f4",
            font=("Segoe UI", 11), relief=tk.FLAT, highlightthickness=1,
            highlightbackground="#45475a", highlightcolor=accent,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
        self.entry.bind("<Return>", self._on_send)
        self.entry.focus_set()

        # Send button
        self.send_btn = tk.Button(
            input_frame, text="Send", bg=accent, fg="#1e1e2e",
            font=("Segoe UI", 10, "bold"), relief=tk.FLAT, padx=16, pady=6,
            activebackground="#b4d0fb", cursor="hand2",
            command=self._on_send,
        )
        self.send_btn.pack(side=tk.LEFT, padx=(6, 0))

        # Voice button
        self.voice_btn = tk.Button(
            input_frame, text="Mic", bg="#45475a", fg="#cdd6f4",
            font=("Segoe UI", 10), relief=tk.FLAT, padx=12, pady=6,
            activebackground="#585b70", cursor="hand2",
        )
        self.voice_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.voice_btn.bind("<ButtonPress-1>", self._on_voice_start)
        self.voice_btn.bind("<ButtonRelease-1>", self._on_voice_stop)

    def _setup_tray(self) -> None:
        """Set up system tray icon."""
        try:
            import pystray
            from PIL import Image, ImageDraw

            # Create a simple icon
            img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rounded_rectangle([4, 4, 60, 60], radius=12, fill=(137, 180, 250))
            draw.text((18, 14), "K", fill=(30, 30, 46), font=None)

            menu = pystray.Menu(
                pystray.MenuItem("Show", self._restore_from_tray, default=True),
                pystray.MenuItem("Quit", self._quit_app),
            )
            self.tray_icon = pystray.Icon("kanna", img, "Kanna", menu)
        except ImportError:
            self.tray_icon = None

    def _setup_hotkey(self) -> None:
        """Register global hotkey Ctrl+Shift+K."""
        try:
            import ctypes
            # Register hotkey: Ctrl+Shift+K (0x4B = K)
            ctypes.windll.user32.RegisterHotKey(None, 1, 0x0006, 0x4B)  # MOD_CONTROL|MOD_SHIFT
            self._poll_hotkey()
        except Exception:
            pass  # Hotkey is a nice-to-have, not critical

    def _poll_hotkey(self) -> None:
        """Poll for the global hotkey message."""
        try:
            import ctypes
            msg = ctypes.wintypes.MSG()
            if ctypes.windll.user32.PeekMessageW(ctypes.byref(msg), None, 0x0312, 0x0312, 1):
                if msg.wParam == 1:
                    self._toggle_window()
        except Exception:
            pass
        self.root.after(100, self._poll_hotkey)

    def _toggle_window(self) -> None:
        if self._hidden:
            self._restore_from_tray()
        else:
            self._minimize_to_tray()

    def _minimize_to_tray(self) -> None:
        self.root.withdraw()
        self._hidden = True
        if self.tray_icon and not self.tray_icon._running:
            threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def _restore_from_tray(self, icon=None, item=None) -> None:
        self.root.after(0, self._do_restore)

    def _do_restore(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self._hidden = False

    def _quit_app(self, icon=None, item=None) -> None:
        if self.tray_icon:
            self.tray_icon.stop()
        self.root.after(0, self.root.destroy)

    # -- Kanna integration ------------------------------------------------

    def _init_kanna(self) -> None:
        try:
            self.kanna = bootstrap()
            self._log("Kanna ready. Type a command or hold the Mic button to speak.", "info")
        except Exception as exc:
            self._log(f"Failed to start Kanna: {exc}", "error")

    def _on_send(self, event=None) -> None:
        text = self.entry.get().strip()
        if not text or not self.kanna:
            return
        self.entry.delete(0, tk.END)
        self._log(f"> {text}", "user")
        threading.Thread(target=self._run_command, args=(text,), daemon=True).start()

    def _run_command(self, text: str) -> None:
        try:
            result = self.kanna.agent_loop().run(text)
            self._log(result.message, "kanna")
        except Exception as exc:
            self._log(f"Error: {exc}", "error")

    def _on_voice_start(self, event=None) -> None:
        if self._recording:
            return
        self._recording = True
        self.voice_btn.configure(bg="#f38ba8", text="...")
        self._log("[listening...]", "info")
        threading.Thread(target=self._record_and_transcribe, daemon=True).start()

    def _on_voice_stop(self, event=None) -> None:
        self._recording = False
        self.voice_btn.configure(bg="#45475a", text="Mic")

    def _record_and_transcribe(self) -> None:
        try:
            from interfaces.voice.listener import listen_once
            text = listen_once(duration=4)
            if text:
                self._log(f"> {text}", "user")
                self._run_command(text)
            else:
                self._log("[no speech detected]", "info")
        except ImportError:
            self._log("Voice not available — install numpy: pip install 'numpy<2'", "error")
        except Exception as exc:
            self._log(f"Voice error: {exc}", "error")
        finally:
            self.root.after(0, lambda: self.voice_btn.configure(bg="#45475a", text="Mic"))
            self._recording = False

    # -- UI helpers -------------------------------------------------------

    def _log(self, text: str, tag: str = "") -> None:
        def _do():
            self.output.configure(state=tk.NORMAL)
            self.output.insert(tk.END, text + "\n", tag)
            self.output.see(tk.END)
            self.output.configure(state=tk.DISABLED)
        self.root.after(0, _do)

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    app = KannaApp()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
