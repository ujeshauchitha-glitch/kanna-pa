"""Single-owner desktop execution and approval bridge. No Tk calls here."""
from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading

from core.agent.session import AgentSession
from core.bootstrap import bootstrap
from core.errors import SandboxViolation


@dataclass
class ApprovalRequest:
    tool_name: str
    args: dict
    reason: str
    answered: threading.Event = field(default_factory=threading.Event)
    allowed: bool = False

    def respond(self, allowed: bool) -> None:
        self.allowed = allowed
        self.answered.set()


class DesktopGate:
    def __init__(self, events, stopped, cancel):
        self.events, self.stopped, self.cancel = events, stopped, cancel

    def approve(self, *, tool_name, args, level, reason):
        request = ApprovalRequest(tool_name, args, reason)
        self.events.put(("approval", request))
        while not request.answered.wait(0.1):
            if self.stopped.is_set() or self.cancel.is_set():
                # Nobody is going to answer a dialog for a task that's
                # being cancelled or an app that's shutting down — deny
                # it and tell the UI so an open dialog closes itself
                # instead of sitting there uselessly.
                request.respond(False)
                self.events.put(("approval_resolved", request))
                break
        return request.allowed and not self.stopped.is_set() and not self.cancel.is_set()


def compose_request(text: str, attachments: list[str]) -> str:
    if not text.strip():
        raise ValueError("Describe what you want to do with the files first.")
    if not attachments:
        return text.strip()
    import json
    return text.strip() + "\n\nUser-selected source paths (read with the appropriate tool):\n" + json.dumps(attachments)


def verified_artifacts(result, sandbox) -> list[str]:
    """Offer only observed, existing, sandbox-contained output files."""
    paths = []
    for outcome in result.outcomes:
        if not outcome.result.success:
            continue
        for raw in outcome.result.files_created + outcome.result.files_modified:
            try:
                path = sandbox.resolve(raw)
                if path.is_file() and str(path) not in paths:
                    paths.append(str(path))
            except (OSError, ValueError, SandboxViolation):
                # A rejected path must not break result display or be offered.
                continue
    return paths


class DesktopWorker:
    def __init__(self, *, factory=bootstrap):
        self.events = queue.Queue()
        self._commands = queue.Queue()
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._factory = factory
        self._lock = threading.Lock()
        self._busy = False
        self.ready = False
        self.thread = threading.Thread(target=self._run, name="kanna-desktop", daemon=True)

    def start(self):
        self.thread.start()

    def submit(self, text: str) -> bool:
        with self._lock:
            if not self.ready or self._busy or self._stop.is_set() or not text.strip():
                return False
            self._busy = True
            self._commands.put(text)
            return True

    def cancel(self) -> bool:
        """Request cancellation of the task currently running. A no-op
        (returns False) if nothing is actually running — there's no
        queue of pending cancellations to apply to a future task."""
        with self._lock:
            if not self._busy:
                return False
            self._cancel.set()
            return True

    def close(self):
        self._stop.set()
        self._commands.put(None)

    def _run(self):
        kanna = None
        session = None
        try:
            kanna = self._factory(gate=DesktopGate(self.events, self._stop, self._cancel))
            session = AgentSession(kanna.db, {"source": "desktop"})
            kanna.event_bus.subscribe("agent.state.", lambda event: self.events.put(
                ("progress", {"state": event.topic.rsplit(".", 1)[-1], **event.payload})))
            self.ready = True
            self.events.put(("ready", {"planner": type(kanna.planner).__name__,
                "model": kanna.settings.llm_model, "roots": [str(p) for p in kanna.sandbox.roots],
                "tools": len(kanna.registry.names()), "db": kanna.db, "settings": kanna.settings}))
            while not self._stop.is_set():
                text = self._commands.get()
                if text is None or self._stop.is_set():
                    break
                self._cancel.clear()
                try:
                    session.log_user_message(text)
                    result = kanna.agent_loop(session_id=session.id).run(text, cancel=self._cancel)
                    session.log_assistant_message(result.message)
                    payload = {"state": result.state.value, "message": result.message,
                               "files": verified_artifacts(result, kanna.sandbox),
                               "steps": [{"tool": o.step.tool_name, "attempts": o.attempts,
                                          "success": o.result.success} for o in result.outcomes]}
                    self.events.put(("result", payload))
                except Exception as exc:
                    self.events.put(("error", str(exc)))
                finally:
                    with self._lock:
                        self._busy = False
                    self.events.put(("idle", None))
        except Exception as exc:
            self.events.put(("startup_error", str(exc)))
        finally:
            self.ready = False
            if kanna is not None:
                try:
                    if session is not None:
                        session.end()
                    # The browser also belongs to this worker thread.
                    from tools.browser import reset_browser_agent
                    reset_browser_agent()
                except Exception as exc:
                    self.events.put(("error", f"Cleanup failed: {exc}"))
                finally:
                    kanna.close()
            self.events.put(("stopped", None))
