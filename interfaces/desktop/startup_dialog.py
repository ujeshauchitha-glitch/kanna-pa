"""Startup-at-login dialog. Enable/disable is only ever a direct result
of clicking a button here — nothing in this module runs on its own.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

from interfaces.desktop.startup import create_startup_backend


class StartupDialog:
    def __init__(self, root, palette, *, backend=None):
        self.p = palette
        self.backend = backend or create_startup_backend()
        self.top = tk.Toplevel(root)
        self.top.title("Start at login — Kanna")
        self.top.geometry("480x260")
        self.top.minsize(420, 220)
        self.top.configure(bg=self.p["BG"])
        self.top.transient(root)
        self._build()
        self._refresh()
        self.top.bind("<Escape>", lambda e: self.top.destroy())
        self.top.focus_set()

    def _label(self, parent, text, size=10, color=None, bold=False, **kw):
        return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=color or self.p["TEXT"],
                        font=("Segoe UI", size, "bold" if bold else "normal"), **kw)

    def _button(self, parent, text, command):
        return tk.Button(parent, text=text, command=command, bg=self.p["PANEL"], fg=self.p["TEXT"],
                         activebackground="#bddfdb", activeforeground=self.p["BG"],
                         disabledforeground="#748496", relief=tk.FLAT, font=("Segoe UI", 10),
                         padx=12, pady=8, cursor="hand2")

    def _build(self):
        self._label(self.top, "Start Kanna at login", 15, bold=True).pack(anchor="w", padx=20, pady=(18, 8))
        self.status = self._label(self.top, "", 10, self.p["MUTED"], wraplength=440, justify=tk.LEFT)
        self.status.pack(anchor="w", padx=20)
        buttons = tk.Frame(self.top, bg=self.p["BG"])
        buttons.pack(anchor="w", padx=20, pady=18)
        self.enable_btn = self._button(buttons, "Enable", self._on_enable)
        self.enable_btn.pack(side=tk.LEFT)
        self.disable_btn = self._button(buttons, "Disable", self._on_disable)
        self.disable_btn.pack(side=tk.LEFT, padx=8)
        self._button(self.top, "Close", self.top.destroy).pack(side=tk.BOTTOM, anchor="e", padx=20, pady=16)

    def _refresh(self):
        if not self.backend.is_supported():
            self.status.configure(text=self.backend.unsupported_reason())
            self.enable_btn.configure(state=tk.DISABLED)
            self.disable_btn.configure(state=tk.DISABLED)
            return
        enabled = self.backend.is_enabled()
        headline = ("Enabled — Kanna launches automatically at login." if enabled else
                    "Disabled — Kanna only launches when you start it yourself.")
        self.status.configure(text=f"{headline}\n\nUses {self.backend.mechanism_description()}.")
        self.enable_btn.configure(state=tk.DISABLED if enabled else tk.NORMAL)
        self.disable_btn.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def _on_enable(self):
        reason = self.backend.enable()
        if reason:
            messagebox.showerror("Could not enable", reason, parent=self.top)
        self._refresh()

    def _on_disable(self):
        reason = self.backend.disable()
        if reason:
            messagebox.showerror("Could not disable", reason, parent=self.top)
        self._refresh()
