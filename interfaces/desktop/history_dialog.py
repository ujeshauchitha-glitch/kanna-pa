"""Task history dialog — browse persisted runs without rerunning them.

Read-only: everything shown comes from `core.agent.history` (built on
the same `plans`/`plan_steps` evidence the agent loop already writes),
queried fresh on open and on Refresh. Nothing here re-executes a step or
mutates the database.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, scrolledtext

from core.agent.history import get_run, list_runs
from interfaces.desktop.files import show_in_folder

_STATUS_BADGE = {"complete": "✓", "failed": "✗", "in_progress": "…"}


class HistoryDialog:
    def __init__(self, root, db, palette):
        self.db = db
        self.p = palette
        self.runs = []
        self.file_paths: list[str] = []

        self.top = tk.Toplevel(root)
        self.top.title("Task history — Kanna")
        self.top.geometry("920x580")
        self.top.minsize(720, 440)
        self.top.configure(bg=self.p["BG"])
        self.top.transient(root)
        self._build()
        self._reload()
        self.top.bind("<Escape>", lambda e: self.top.destroy())
        self.top.focus_set()

    def _label(self, parent, text, size=10, color=None, bold=False, **kw):
        return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=color or self.p["TEXT"],
                        font=("Segoe UI", size, "bold" if bold else "normal"), **kw)

    def _button(self, parent, text, command):
        return tk.Button(parent, text=text, command=command, bg=self.p["PANEL"], fg=self.p["TEXT"],
                         activebackground="#bddfdb", activeforeground=self.p["BG"], relief=tk.FLAT,
                         font=("Segoe UI", 10), padx=10, pady=6, cursor="hand2")

    def _build(self):
        bar = tk.Frame(self.top, bg=self.p["BG"])
        bar.pack(fill=tk.X, padx=16, pady=(14, 6))
        self._label(bar, "Task history", 15, bold=True).pack(side=tk.LEFT)
        self._button(bar, "Close", self.top.destroy).pack(side=tk.RIGHT)
        self._button(bar, "Refresh", self._reload).pack(side=tk.RIGHT, padx=6)

        body = tk.Frame(self.top, bg=self.p["BG"])
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 14))

        left = tk.Frame(body, bg=self.p["PANEL"], width=280)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        left.pack_propagate(False)
        self._label(left, "RUNS", 9, self.p["MUTED"], True).pack(anchor="w", padx=10, pady=(10, 4))
        self.runs_list = tk.Listbox(left, bg=self.p["FIELD"], fg=self.p["TEXT"],
            selectbackground="#34584f", selectforeground=self.p["TEXT"], relief=tk.FLAT,
            borderwidth=0, highlightthickness=0, font=("Segoe UI", 10), exportselection=False)
        self.runs_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.runs_list.bind("<<ListboxSelect>>", self._on_select_run)

        right = tk.Frame(body, bg=self.p["BG"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.detail = scrolledtext.ScrolledText(right, wrap=tk.WORD, state=tk.DISABLED,
            bg=self.p["FIELD"], fg=self.p["TEXT"], font=("Segoe UI", 10), relief=tk.FLAT,
            padx=14, pady=10, highlightthickness=0)
        for name, color in [("head", self.p["ACCENT"]), ("ok", self.p["TEXT"]),
                            ("failed", self.p["ERROR"]), ("muted", self.p["MUTED"])]:
            self.detail.tag_config(name, foreground=color, spacing1=4, spacing3=6)
        self.detail.pack(fill=tk.BOTH, expand=True)

        self._label(right, "FILES", 9, self.p["MUTED"], True).pack(anchor="w", pady=(8, 0))
        self.files_list = tk.Listbox(right, height=4, bg=self.p["PANEL"], fg=self.p["TEXT"],
            selectbackground="#34584f", selectforeground=self.p["TEXT"], relief=tk.FLAT,
            borderwidth=0, highlightthickness=0, font=("Segoe UI", 10), exportselection=False)
        self.files_list.pack(fill=tk.X, pady=(4, 4))
        actions = tk.Frame(right, bg=self.p["BG"])
        actions.pack(fill=tk.X)
        self._button(actions, "Copy path", self._copy_selected_path).pack(side=tk.LEFT)
        self._button(actions, "Show folder", self._show_selected_folder).pack(side=tk.LEFT, padx=6)

    def _reload(self):
        previously_selected = self._selected_run_id()
        self.runs = list_runs(self.db)
        self.runs_list.delete(0, tk.END)
        for run in self.runs:
            badge = _STATUS_BADGE.get(run.status, "?")
            excerpt = run.request.replace("\n", " ")
            excerpt = excerpt if len(excerpt) <= 44 else excerpt[:41] + "…"
            when = run.created_at[:16].replace("T", " ")
            self.runs_list.insert(tk.END, f"{badge}  {when}  {excerpt}")
        if not self.runs:
            self._show_detail_text("No task history yet. Runs appear here after you run a task.", "muted")
            self.files_list.delete(0, tk.END)
            self.file_paths = []
            return
        target_index = 0
        if previously_selected:
            for i, run in enumerate(self.runs):
                if run.id == previously_selected:
                    target_index = i
                    break
        self.runs_list.selection_set(target_index)
        self._show_run(self.runs[target_index].id)

    def _selected_run_id(self):
        sel = self.runs_list.curselection()
        return self.runs[sel[0]].id if sel and sel[0] < len(self.runs) else None

    def _on_select_run(self, event=None):
        run_id = self._selected_run_id()
        if run_id:
            self._show_run(run_id)

    def _show_run(self, run_id):
        detail = get_run(self.db, run_id)
        self.files_list.delete(0, tk.END)
        self.file_paths = []
        if detail is None:
            self._show_detail_text("This run is no longer in the database.", "muted")
            return
        lines = [
            (f"{detail.request}\n", "head"),
            (f"{detail.status}  ·  started {detail.created_at}  ·  "
             f"updated {detail.updated_at}\n\n", "muted"),
        ]
        for i, step in enumerate(detail.steps, start=1):
            tag = "ok" if step.status == "ok" else ("failed" if step.status == "failed" else "muted")
            attempts = f", {step.attempts} attempt(s)" if step.attempts else ""
            lines.append((f"{i}. {step.tool_name}  [{step.status}{attempts}]\n", tag))
            lines.append((f"   {step.message}\n", "muted"))
            for f in step.files:
                prefix = "✓  " if f.exists else "✗ missing  "
                self.files_list.insert(tk.END, prefix + f.path)
                self.file_paths.append(f.path)
        self._render_detail(lines)

    def _render_detail(self, lines):
        self.detail.configure(state=tk.NORMAL)
        self.detail.delete("1.0", tk.END)
        for text, tag in lines:
            self.detail.insert(tk.END, text, tag)
        self.detail.configure(state=tk.DISABLED)

    def _show_detail_text(self, text, tag):
        self._render_detail([(text, tag)])

    def _selected_file(self):
        sel = self.files_list.curselection()
        return self.file_paths[sel[0]] if sel and sel[0] < len(self.file_paths) else None

    def _copy_selected_path(self):
        path = self._selected_file()
        if path:
            self.top.clipboard_clear()
            self.top.clipboard_append(path)

    def _show_selected_folder(self):
        path = self._selected_file()
        if not path:
            return
        try:
            show_in_folder(path)
        except OSError as exc:
            messagebox.showerror("Cannot open folder", str(exc), parent=self.top)
