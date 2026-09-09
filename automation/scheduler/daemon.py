"""A foreground loop around `Scheduler.tick()` — the actual "set and
forget" mode.

Phase 1 shipped only `tick()`, meant to be invoked once per call by an
external scheduler (cron, a systemd timer, a person). This adds a
process that loops, ticking on a fixed interval, until told to stop —
without requiring any *other* scheduler to drive it. Nothing about
*what* runs changes, only *when this process itself calls `tick()`
next* — every actual job execution still goes through the exact same
`Scheduler`/`on_due` path as before.

`sleep` is injectable specifically so tests never depend on wall-clock
time — the same discipline `finance`/`scheduler`'s pure functions
already apply to `now`. See `docs/SCHEDULER.md` for how to run this as
a systemd service instead of a cron-invoked `kanna scheduler tick`.
"""
from __future__ import annotations

import signal
import time
from collections.abc import Callable

from automation.scheduler.scheduler import Scheduler

TickCallback = Callable[[list[dict]], None]


class SchedulerDaemon:
    def __init__(self, scheduler: Scheduler, *, interval_seconds: float = 60.0,
                 sleep: Callable[[float], None] = time.sleep,
                 on_tick: TickCallback | None = None) -> None:
        self.scheduler = scheduler
        self.interval_seconds = interval_seconds
        self._sleep = sleep
        self._on_tick = on_tick
        self._stop = False

    def stop(self) -> None:
        """Request the run loop exit after its current tick. Safe to call
        from a signal handler (see `install_signal_handlers`) — this only
        sets a flag, it never touches I/O itself."""
        self._stop = True

    def run_forever(self) -> None:
        """Tick every `interval_seconds`, forever, until `stop()` is called."""
        self._stop = False
        while not self._stop:
            self._tick_once()
            if self._stop:
                break
            self._sleep(self.interval_seconds)

    def run_n_ticks(self, n: int) -> list[list[dict]]:
        """Run exactly `n` ticks, sleeping between them (not after the
        last) — the bounded, testable alternative to `run_forever()`.
        Returns each tick's outcomes, in order.
        """
        results: list[list[dict]] = []
        for i in range(n):
            results.append(self._tick_once())
            if i < n - 1:
                self._sleep(self.interval_seconds)
        return results

    def _tick_once(self) -> list[dict]:
        outcomes = self.scheduler.tick()
        if self._on_tick is not None:
            self._on_tick(outcomes)
        return outcomes


def install_signal_handlers(daemon: SchedulerDaemon) -> None:
    """Wire SIGINT/SIGTERM to a clean `daemon.stop()` instead of an abrupt
    kill — so a job mid-execution finishes its current tick rather than
    being cut off, and `run_forever()` returns normally. Not called by
    `SchedulerDaemon` itself (a library shouldn't install process-wide
    signal handlers on construction) — the CLI entry point
    (`kanna scheduler daemon`) calls this explicitly.
    """
    def _handler(signum, frame):  # noqa: ARG001 - required signal handler signature
        daemon.stop()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
