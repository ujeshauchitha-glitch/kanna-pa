"""Native task workspace. All Tk operations stay on the main thread."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from core.permissions.sandbox import Sandbox
from interfaces.desktop.worker import DesktopWorker, compose_request

BG = "#10151d"
PANEL = "#19212d"
FIELD = "#111923"
TEXT = "#e7edf5"
MUTED = "#a5b3c6"
ACCENT = "#9ce6cf"
ERROR = "#ffb0b0"


class KannaApp:
    def __init__(self, *, worker=None, integrations=True):
        self.root = tk.Tk()
        self.root.title("Kanna — Task workspace")
        self.root.geometry("1120x780")
        self.root.minsize(860, 640)
        self.root.configure(bg=BG)
        self.worker = worker or DesktopWorker()
        self.attachments = []
        self.files = []
        self.info = {}
        self.busy = False
        self.recording = False
        self.closing = False
        self.last_result = ""
        self.tray_icon = None
        self._hotkey = False
        self._build_ui()
        self.root.bind("<Configure>", self._resize)
        if integrations:
            self._setup_tray()
            self._setup_hotkey()
        self.root.protocol("WM_DELETE_WINDOW", self._hide)
        self.root.after(80, self._poll)
        self.worker.start()

    def _label(self, parent, text, size=10, color=TEXT, bold=False, **kwargs):
        return tk.Label(parent, text=text, bg=parent.cget("bg"), fg=color,
                        font=("Segoe UI", size, "bold" if bold else "normal"), **kwargs)

    def _button(self, parent, text, command, primary=False):
        return tk.Button(parent, text=text, command=command, bg=ACCENT if primary else PANEL,
                         fg=BG if primary else TEXT, activebackground="#bddfdb",
                         activeforeground=BG, disabledforeground="#748496", relief=tk.FLAT,
                         font=("Segoe UI", 10, "bold" if primary else "normal"),
                         padx=12, pady=8, cursor="hand2", takefocus=True)

    def _build_ui(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Kanna.Horizontal.TProgressbar", background=ACCENT,
                        troughcolor=PANEL, borderwidth=0, thickness=4)
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill=tk.X, padx=24, pady=(20, 14))
        self._label(top, "K /", 22, ACCENT, True).pack(side=tk.LEFT)
        self._label(top, "  Kanna", 22, bold=True).pack(side=tk.LEFT)
        self._label(top, "   YOUR TASK WORKSPACE", 9, MUTED).pack(side=tk.LEFT, pady=(7, 0))
        self._button(top, "Quit", self._quit).pack(side=tk.RIGHT)
        self._button(top, "Connection & access", self._show_info).pack(side=tk.RIGHT, padx=8)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 20))
        side = tk.Frame(body, bg=PANEL, width=212)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 18))
        side.pack_propagate(False)
        self._label(side, "START SOMETHING", 9, MUTED, True).pack(anchor="w", padx=16, pady=(20, 12))
        starters = [
            ("Read a document", "Read the attached document and summarize its key points. Cite the source and flag anything unclear."),
            ("Create a report", "Read the attached source and create a structured PDF report in the workspace. Verify it and return the saved path."),
            ("Prepare an assignment", "Read the attached assignment and reference material. Answer the questions, flag unresolved items, and save a DOCX in the workspace."),
            ("Check my spending", "How much did I spend this month?"),
            ("Explore workspace", "list files in ."),
        ]
        for title, prompt in starters:
            self._button(side, title, lambda p=prompt: self._draft(p)).pack(fill=tk.X, padx=10, pady=3)
        self._label(side, "Task starters fill your draft.\nReview it before running.", 9, MUTED,
                    justify=tk.LEFT).pack(anchor="w", padx=16, pady=(12, 24))
        self._label(side, "CURRENT CONNECTION", 9, MUTED, True).pack(anchor="w", padx=16)
        self.connection = self._label(side, "Starting…", 10, wraplength=175, justify=tk.LEFT)
        self.connection.pack(anchor="w", padx=16, pady=(8, 12))
        self.access_note = self._label(side, "Work stays in your configured\nfolders. Actions needing review\nask for approval here.", 9, MUTED, justify=tk.LEFT)
        self.access_note.pack(side=tk.BOTTOM, anchor="w", padx=16, pady=20)

        main = tk.Frame(body, bg=BG)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._label(main, "What would you like to get done?", 20, bold=True).pack(anchor="w")
        self.intro = self._label(main, "Describe the result. Add sources. Follow the work as it happens.", 10, MUTED)
        self.intro.pack(anchor="w", pady=(5, 14))
        composer = tk.Frame(main, bg=PANEL, padx=12, pady=12)
        self.composer = composer
        composer.pack(fill=tk.X)
        self.entry = tk.Text(composer, height=4, width=30, wrap=tk.WORD, bg=FIELD, fg=TEXT,
                             insertbackground=ACCENT, relief=tk.FLAT, padx=10, pady=10,
                             font=("Segoe UI", 11), undo=True, highlightthickness=1,
                             highlightbackground="#354458", highlightcolor=ACCENT)
        self.entry.pack(fill=tk.X)
        self.entry.bind("<Control-Return>", self._on_send)
        actions = tk.Frame(composer, bg=PANEL)
        actions.pack(fill=tk.X, pady=(10, 0))
        self.attach_btn = self._button(actions, "+ Add source", self._attach)
        self.attach_btn.pack(side=tk.LEFT)
        self.voice_btn = self._button(actions, "Record 4 seconds", self._voice)
        self.voice_btn.pack(side=tk.LEFT, padx=6)
        self.send_btn = self._button(actions, "Run task  →", self._on_send, True)
        self.send_btn.pack(side=tk.RIGHT)
        self.send_btn.configure(state=tk.DISABLED)
        self.attach_btn.configure(state=tk.DISABLED)
        self._label(composer, "Ctrl+Enter to run · Voice fills the draft for review (Google transcription)", 9, MUTED).pack(anchor="w", pady=(8, 0))
        self.sources = self._label(composer, "No sources attached", 9, MUTED, anchor="w")
        self.sources.pack(fill=tk.X, pady=(6, 0))
        composer.bind("<Configure>", lambda e: self.sources.configure(wraplength=max(200, e.width - 30)))
        self.clear_sources = self._button(composer, "Remove sources", self._remove_sources)

        state_row = tk.Frame(main, bg=BG)
        state_row.pack(fill=tk.X, pady=(14, 6))
        self.status = self._label(state_row, "Starting Kanna…", 10, ACCENT, True)
        self.status.pack(side=tk.LEFT)
        self._button(state_row, "Copy result", self._copy_result).pack(side=tk.RIGHT)
        self.progress = ttk.Progressbar(main, mode="indeterminate", style="Kanna.Horizontal.TProgressbar")
        self.progress.pack(fill=tk.X, pady=(0, 8))
        self.output = scrolledtext.ScrolledText(main, wrap=tk.WORD, state=tk.DISABLED,
            bg=FIELD, fg=TEXT, font=("Segoe UI", 11), relief=tk.FLAT, padx=16, pady=12,
            height=6, width=40, highlightthickness=0)
        for name, color in [("user", ACCENT), ("error", ERROR), ("info", MUTED), ("kanna", TEXT)]:
            self.output.tag_config(name, foreground=color, spacing1=6, spacing3=8)
        self._log("Your workspace is ready for a task. Choose a starter or write your own request.\nResults and execution details will appear here.", "info")
        footer = tk.Frame(main, bg=BG)
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        self._label(footer, "OUTPUT FILES", 9, MUTED, True).pack(anchor="w", pady=(12, 4))
        self.artifacts = tk.Listbox(footer, height=2, bg=PANEL, fg=TEXT, selectbackground="#34584f",
                                   selectforeground=TEXT, relief=tk.FLAT, borderwidth=0, highlightthickness=0,
                                   font=("Segoe UI", 10), exportselection=False)
        self.artifacts.pack(fill=tk.X)
        file_actions = tk.Frame(footer, bg=BG)
        file_actions.pack(fill=tk.X, pady=(4, 0))
        self._button(file_actions, "Copy path", self._copy_path).pack(side=tk.LEFT)
        self._button(file_actions, "Show folder", self._show_folder).pack(side=tk.LEFT, padx=6)
        self.output.pack(fill=tk.BOTH, expand=True)
        self.entry.focus_set()

    def _draft(self, text):
        if self.busy or self.recording or self.closing:
            return
        self.entry.delete("1.0", tk.END)
        self.entry.insert("1.0", text)
        self.entry.focus_set()

    def _resize(self, event):
        if event.widget is not self.root:
            return
        compact = event.height < 720
        self.entry.configure(height=2 if compact else 4)
        self.artifacts.configure(height=1 if compact else 2)
        if compact:
            self.intro.pack_forget()
            self.access_note.pack_forget()
        else:
            self.intro.pack(anchor="w", pady=(5, 14), before=self.composer)
            self.access_note.pack(side=tk.BOTTOM, anchor="w", padx=16, pady=20)

    def _attach(self):
        if not self.info or self.busy:
            return
        paths = filedialog.askopenfilenames(parent=self.root, title="Add source documents",
            initialdir=self.info["roots"][0], filetypes=[("Documents", "*.txt *.md *.pdf *.docx *.png *.jpg *.jpeg"), ("All files", "*")])
        sandbox = Sandbox(self.info["roots"])
        for raw in paths:
            try:
                path = str(sandbox.resolve(raw))
                if path not in self.attachments:
                    self.attachments.append(path)
            except Exception as exc:
                messagebox.showerror("Source outside workspace", f"{exc}\n\nCopy the source into a configured workspace folder first.", parent=self.root)
        self._source_label()

    def _remove_sources(self):
        if not self.busy:
            self.attachments.clear()
            self._source_label()

    def _source_label(self):
        names = [Path(p).name for p in self.attachments]
        summary = " · ".join(name if len(name) < 35 else name[:32] + "…" for name in names[:2])
        if len(names) > 2:
            summary += f" · +{len(names) - 2} more"
        self.sources.configure(text=summary or "No sources attached")
        if self.attachments:
            self.clear_sources.pack(anchor="w")
        else:
            self.clear_sources.pack_forget()

    def _on_send(self, event=None):
        if self.busy or self.recording or self.closing:
            return "break"
        text = self.entry.get("1.0", "end-1c").strip()
        if not text:
            self.status.configure(text="Describe a task first", fg=MUTED)
            self.entry.focus_set()
            return "break"
        request = compose_request(text, self.attachments)
        if not self.worker.submit(request):
            self.status.configure(text="Kanna is not ready yet", fg=MUTED)
            return "break"
        self._log(text, "user")
        self.files = []
        self.artifacts.delete(0, tk.END)
        self.last_result = ""
        self._set_busy(True)
        return "break"

    def _set_busy(self, busy):
        self.busy = busy
        state = tk.DISABLED if busy or self.closing else tk.NORMAL
        for widget in (self.entry, self.send_btn, self.attach_btn, self.voice_btn, self.clear_sources):
            widget.configure(state=state)
        if busy:
            self.progress.start(14)
        else:
            self.progress.stop()

    def _poll(self):
        for _ in range(100):
            try:
                kind, payload = self.worker.events.get_nowait()
            except queue.Empty:
                break
            if kind == "ready":
                self.info = payload
                limited = payload["planner"] == "RuleBasedPlanner"
                self.connection.configure(text=("Basic commands only\nLLM not configured" if limited else payload["model"]) + f"\n{payload['tools']} registered tools")
                self.status.configure(text="Ready · basic mode" if limited else "Ready", fg=ACCENT)
                self._set_busy(False)
                if limited:
                    self._log("Basic mode supports simple file and spending commands. Configure an LLM for document authoring and open-ended tasks. See Connection & access.", "info")
            elif kind == "show":
                self._restore()
            elif kind == "tray_failed":
                self.tray_icon = None
            elif kind == "progress":
                label = payload["state"].replace("_", " ").capitalize()
                tool = payload.get("tool") or payload.get("step")
                self.status.configure(text=f"{label}{' · ' + tool if tool else ''}", fg=ACCENT)
                if payload["state"] in ("executing", "correcting", "verifying"):
                    self._log(f"{label}: {tool or ''}", "info")
            elif kind == "approval":
                self._approve(payload)
            elif kind == "result":
                self.last_result = payload["message"]
                self.status.configure(text={"complete": "Completed", "failed": "Task failed", "blocked": "Needs attention"}.get(payload["state"], payload["state"]), fg=ACCENT if payload["state"] == "complete" else ERROR)
                self._log(payload["message"], "kanna" if payload["state"] == "complete" else "error")
                self.files = payload["files"]
                for path in self.files:
                    self.artifacts.insert(tk.END, path)
                if self.files:
                    self.artifacts.selection_set(0)
            elif kind in ("error", "startup_error"):
                self.status.configure(text="Could not start" if kind == "startup_error" else "Task failed", fg=ERROR)
                self._log(str(payload), "error")
            elif kind == "idle":
                self._set_busy(False)
            elif kind == "transcript":
                self.recording = False
                self._set_busy(False)
                self.voice_btn.configure(text="Record 4 seconds")
                if payload:
                    self.entry.insert(tk.END, ("\n" if self.entry.get("1.0", "end-1c") else "") + payload)
                    self.status.configure(text="Transcript ready · review and run", fg=ACCENT)
                else:
                    self.status.configure(text="No speech detected", fg=MUTED)
            elif kind == "voice_error":
                self.recording = False
                self._set_busy(False)
                self.voice_btn.configure(text="Record 4 seconds")
                self.status.configure(text="Voice unavailable · you can still type", fg=ERROR)
                self._log(payload, "error")
            elif kind == "stopped" and self.closing:
                self._destroy()
                return
        self.root.after(80, self._poll)

    def _approve(self, request):
        if self.closing:
            request.respond(False)
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Review action — Kanna")
        dialog.geometry("620x420")
        dialog.configure(bg=PANEL)
        dialog.transient(self.root)
        self._label(dialog, f"Allow {request.tool_name}?", 16, bold=True).pack(anchor="w", padx=20, pady=16)
        self._label(dialog, request.reason, 10, MUTED, wraplength=570).pack(anchor="w", padx=20)
        details = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, height=12, bg=FIELD, fg=TEXT)
        details.pack(fill=tk.BOTH, expand=True, padx=20, pady=12)
        details.insert("1.0", json.dumps(request.args, indent=2, ensure_ascii=False))
        details.configure(state=tk.DISABLED)
        def answer(value):
            request.respond(value)
            dialog.destroy()
        buttons = tk.Frame(dialog, bg=PANEL)
        buttons.pack(fill=tk.X, padx=20, pady=(0, 16))
        deny = self._button(buttons, "Deny", lambda: answer(False))
        deny.pack(side=tk.RIGHT)
        self._button(buttons, "Allow once", lambda: answer(True), True).pack(side=tk.RIGHT, padx=10)
        dialog.protocol("WM_DELETE_WINDOW", lambda: answer(False))
        dialog.bind("<Escape>", lambda e: answer(False))
        dialog.grab_set()
        deny.focus_set()

    def _voice(self):
        if self.busy or self.recording or self.closing:
            return
        self.recording = True
        self._set_busy(True)
        self.voice_btn.configure(text="Listening…")
        self.status.configure(text="Recording 4 seconds, then transcribing…", fg=ACCENT)
        def record():
            try:
                from interfaces.voice.listener import listen_once
                self.worker.events.put(("transcript", listen_once(duration=4)))
            except Exception as exc:
                self.worker.events.put(("voice_error", f"{exc}\nVoice requires the [voice] extra, a microphone, and network access for Google transcription."))
        threading.Thread(target=record, daemon=True).start()

    def _show_info(self):
        roots = "\n".join(self.info.get("roots", [])) or "Starting…"
        messagebox.showinfo("Connection & access", f"Planner: {self.info.get('planner', 'Starting…')}\nConfigured model: {self.info.get('model', '—')}\n\nWorkspace folders:\n{roots}\n\nLLM configuration: KANNA_LLM_PROVIDER and KANNA_LLM_MODEL, or ~/.kanna/config.toml. Restart after changes. A configured model is not a guarantee that its server is reachable.", parent=self.root)

    def _copy_result(self):
        if self.last_result:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.last_result)

    def _selected_path(self):
        selected = self.artifacts.curselection()
        return self.files[selected[0]] if selected else None

    def _copy_path(self):
        path = self._selected_path()
        if path:
            self.root.clipboard_clear()
            self.root.clipboard_append(path)

    def _show_folder(self):
        path = self._selected_path()
        if path:
            try:
                parent = Path(path).parent
                if sys.platform == "win32":
                    os.startfile(str(parent))
                else:
                    subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(parent)])
            except OSError as exc:
                messagebox.showerror("Cannot open folder", str(exc), parent=self.root)

    def _log(self, text, tag="info"):
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, text + "\n", tag)
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)

    def _setup_tray(self):
        try:
            import pystray
            from PIL import Image, ImageDraw
            image = Image.new("RGB", (64, 64), BG)
            ImageDraw.Draw(image).text((24, 22), "K", fill=ACCENT)
            self.tray_icon = pystray.Icon("kanna", image, "Kanna", pystray.Menu(
                pystray.MenuItem("Show", lambda *a: self.worker.events.put(("show", None)), default=True)))
            def tray():
                try:
                    self.tray_icon.run()
                except Exception:
                    self.worker.events.put(("tray_failed", None))
            threading.Thread(target=tray, daemon=True).start()
        except Exception:
            self.tray_icon = None

    def _setup_hotkey(self):
        if sys.platform == "win32":
            import ctypes
            self._hotkey = bool(ctypes.windll.user32.RegisterHotKey(None, 1, 0x0006, 0x4B))
            if self._hotkey:
                self._poll_hotkey()

    def _poll_hotkey(self):
        import ctypes
        from ctypes import wintypes
        msg = wintypes.MSG()
        if ctypes.windll.user32.PeekMessageW(ctypes.byref(msg), None, 0x0312, 0x0312, 1):
            self._restore()
        if not self.closing:
            self.root.after(100, self._poll_hotkey)

    def _hide(self):
        # Always retrievable from the taskbar even if optional tray setup fails.
        self.root.iconify()

    def _restore(self):
        self.root.deiconify()
        self.root.lift()

    def _quit(self):
        if self.closing:
            return
        if self.busy and not messagebox.askyesno("Quit after current work?", "Kanna will wait for the current operation to finish before closing. This does not cancel or undo executed actions.", parent=self.root):
            return
        self.closing = True
        self._set_busy(True)
        self.status.configure(text="Finishing current operation before closing…", fg=MUTED)
        self.worker.close()
        if not self.worker.thread.is_alive():
            self._destroy()

    def _destroy(self):
        if self.tray_icon:
            self.tray_icon.stop()
        if self._hotkey:
            import ctypes
            ctypes.windll.user32.UnregisterHotKey(None, 1)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    try:
        KannaApp().run()
    except tk.TclError as exc:
        print(f"Desktop unavailable: {exc}. Install Python with Tcl/Tk support.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
