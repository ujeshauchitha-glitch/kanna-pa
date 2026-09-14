"""Cross-platform single-instance guard and focus-existing-window signal.

Binds a TCP listener on ``127.0.0.1`` on a port derived deterministically
from the resolved ``KANNA_HOME`` path, so exactly one desktop app can hold
it per configured home on a given machine. A second launch fails to bind,
sends one short message to the first instance instead, and returns without
ever constructing a worker or a window — this is what actually prevents
duplicate instances and duplicate execution, not just a UI-level check.

A crashed or killed instance releases the OS-level port automatically, so
there is no stale lock file to detect or clean up by hand. Using a loopback
TCP port (rather than a lock file) also means this works identically
whether Kanna is launched from inside the repository, from an arbitrary
working directory, or via a shortcut whose target path contains spaces —
nothing here depends on argv[0], cwd, or the on-disk repo location at all.
"""
from __future__ import annotations

import hashlib
import socket
import threading
from pathlib import Path
from typing import Callable

_HOST = "127.0.0.1"
_MAGIC = b"KANNA-DESKTOP-SHOW\n"
# IANA "dynamic/private" range, picked at random per KANNA_HOME so
# different configured homes (or two tests' separate tmp_path homes)
# don't collide with each other or with a real running instance.
_PORT_BASE = 49200
_PORT_SPAN = 800


def port_for(home: Path) -> int:
    digest = hashlib.sha256(str(Path(home).resolve()).encode("utf-8")).digest()
    return _PORT_BASE + (int.from_bytes(digest[:2], "big") % _PORT_SPAN)


class SingleInstanceGuard:
    """Bind on construction; `.acquired` tells you whether this process
    is the only instance for this KANNA_HOME. Construction never blocks.
    """

    def __init__(self, home: Path):
        self.port = port_for(home)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self.acquired = False
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((_HOST, self.port))
            sock.listen(4)
            sock.settimeout(0.5)
            self._sock = sock
            self.acquired = True
        except OSError:
            sock.close()

    def notify_existing(self, timeout: float = 1.0) -> bool:
        """Ask the instance already holding this port to show its window.

        Returns whether the message was actually delivered (a connection
        refused/timed out means the port is stuck in some other state,
        not necessarily a live, responsive Kanna instance).
        """
        try:
            with socket.create_connection((_HOST, self.port), timeout=timeout) as conn:
                conn.sendall(_MAGIC)
            return True
        except OSError:
            return False

    def start(self, on_show: Callable[[], None]) -> None:
        """Begin serving focus-existing-window requests. No-op if this
        process did not win the bind (nothing to serve)."""
        if not self.acquired or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._serve, args=(on_show,), name="kanna-singleton", daemon=True)
        self._thread.start()

    def _serve(self, on_show: Callable[[], None]) -> None:
        sock = self._sock
        while sock is not None:
            try:
                conn, _addr = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                data = conn.recv(len(_MAGIC))
                if data == _MAGIC:
                    on_show()
            except OSError:
                pass
            finally:
                conn.close()

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
