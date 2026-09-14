"""Connection & access dialog: read-only workspace info (planner, tool
count, sandboxed folders), plus an editable LLM provider/model form that
writes to config.toml. Saving never takes effect on the already-running
worker — bootstrap() only runs once per launch — so this says exactly
that rather than pretending a live reload happened.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

from core.config import paths
from core.config.settings import update_config_file

_PROVIDERS = ("litellm", "anthropic", "rule_based")


class ConnectionDialog:
    def __init__(self, root, info, settings, palette):
        self.p = palette
        self.info = info or {}
        self.settings = settings
        self.top = tk.Toplevel(root)
        self.top.title("Connection & access — Kanna")
        self.top.geometry("560x560")
        self.top.minsize(480, 480)
        self.top.configure(bg=self.p["BG"])
        self.top.transient(root)
        self._build()
        self.top.bind("<Escape>", lambda e: self.top.destroy())
        self.top.focus_set()

    def _label(self, parent, text, size=10, color=None, bold=False, **kw):
        return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=color or self.p["TEXT"],
                        font=("Segoe UI", size, "bold" if bold else "normal"), **kw)

    def _button(self, parent, text, command, primary=False):
        return tk.Button(parent, text=text, command=command,
                         bg=self.p["ACCENT"] if primary else self.p["PANEL"],
                         fg=self.p["BG"] if primary else self.p["TEXT"],
                         activebackground="#bddfdb", activeforeground=self.p["BG"], relief=tk.FLAT,
                         font=("Segoe UI", 10, "bold" if primary else "normal"), padx=12, pady=8,
                         cursor="hand2")

    def _entry(self, parent, value):
        entry = tk.Entry(parent, bg=self.p["FIELD"], fg=self.p["TEXT"],
                         insertbackground=self.p["ACCENT"], relief=tk.FLAT, font=("Segoe UI", 10),
                         highlightthickness=1, highlightbackground="#354458",
                         highlightcolor=self.p["ACCENT"])
        entry.insert(0, value)
        return entry

    def _build(self):
        self._label(self.top, "Connection & access", 15, bold=True).pack(anchor="w", padx=20, pady=(18, 8))
        roots = "\n".join(self.info.get("roots", [])) or "Starting…"
        self._label(self.top,
            f"Planner: {self.info.get('planner', 'Starting…')}\n"
            f"{self.info.get('tools', 0)} registered tools\n\n"
            f"Workspace folders (work stays inside these):\n{roots}",
            10, self.p["MUTED"], justify=tk.LEFT, wraplength=520).pack(anchor="w", padx=20)

        form = tk.Frame(self.top, bg=self.p["BG"])
        form.pack(fill=tk.X, padx=20, pady=(20, 6))
        self._label(form, "LLM PROVIDER SETTINGS", 9, self.p["MUTED"], True).pack(anchor="w", pady=(0, 8))

        self._label(form, "Provider", 9, self.p["MUTED"]).pack(anchor="w")
        current_provider = self.settings.llm_provider if self.settings else _PROVIDERS[0]
        self.provider_var = tk.StringVar(value=current_provider)
        provider_menu = tk.OptionMenu(form, self.provider_var, *_PROVIDERS)
        provider_menu.configure(bg=self.p["FIELD"], fg=self.p["TEXT"], relief=tk.FLAT,
                                highlightthickness=1, highlightbackground="#354458",
                                activebackground=self.p["PANEL"], activeforeground=self.p["TEXT"])
        provider_menu["menu"].configure(bg=self.p["FIELD"], fg=self.p["TEXT"])
        provider_menu.pack(anchor="w", fill=tk.X, pady=(2, 10))

        self._label(form, "Model", 9, self.p["MUTED"]).pack(anchor="w")
        self.model_entry = self._entry(form, self.settings.llm_model if self.settings else "")
        self.model_entry.pack(fill=tk.X, pady=(2, 10))

        self._label(form, "Fallback models (comma-separated, litellm only)", 9, self.p["MUTED"]).pack(anchor="w")
        self.fallback_entry = self._entry(
            form, self.settings.llm_fallback_models if self.settings else "")
        self.fallback_entry.pack(fill=tk.X, pady=(2, 10))

        self._label(form, "Timeout (seconds)", 9, self.p["MUTED"]).pack(anchor="w")
        self.timeout_entry = self._entry(
            form, str(self.settings.llm_timeout_seconds) if self.settings else "120")
        self.timeout_entry.pack(fill=tk.X, pady=(2, 10))

        self.feedback = self._label(self.top, "", 9, self.p["MUTED"], wraplength=520, justify=tk.LEFT)
        self.feedback.pack(anchor="w", padx=20, pady=(6, 0))

        buttons = tk.Frame(self.top, bg=self.p["BG"])
        buttons.pack(fill=tk.X, padx=20, pady=16, side=tk.BOTTOM)
        self._button(buttons, "Close", self.top.destroy).pack(side=tk.RIGHT)
        self._button(buttons, "Save", self._on_save, True).pack(side=tk.RIGHT, padx=8)

    def _on_save(self):
        timeout_text = self.timeout_entry.get().strip()
        try:
            timeout = int(timeout_text)
            if timeout <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid timeout", "Timeout must be a positive whole number of seconds.", parent=self.top)
            return
        model = self.model_entry.get().strip()
        if not model:
            messagebox.showerror("Missing model", "Enter a model name.", parent=self.top)
            return
        try:
            config_path = paths.config_path()
            update_config_file(config_path, {
                "llm_provider": self.provider_var.get(),
                "llm_model": model,
                "llm_fallback_models": self.fallback_entry.get().strip(),
                "llm_timeout_seconds": timeout,
            })
        except OSError as exc:
            messagebox.showerror("Could not save", str(exc), parent=self.top)
            return
        self.feedback.configure(
            text=f"Saved to {config_path}. Restart Kanna (Quit, then launch again) to use it.",
            fg=self.p["ACCENT"])
