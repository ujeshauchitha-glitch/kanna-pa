# Scheduler

## Status: real, testable, and now actually runnable two ways

`automation/scheduler/schedule.py` declares the pure, deterministic core: `Schedule` (once/interval/
weekly), `compute_next_run()`, `is_due()` — no I/O, no wall clock dependency baked in, trivially unit
tested by passing an explicit `now`. `automation/scheduler/store.py::SchedulerStore` persists
schedules and their run history to SQLite (`schedules`/`jobs` tables). `automation/scheduler/
scheduler.py::Scheduler.tick()` finds every due schedule, hands each to an `on_due` callback (the CLI
wires this to `kanna.agent_loop().run(schedule.config["request"])` — a scheduled job is a real agent
run, not a special code path), and records the outcome.

What Phase 1 shipped was `tick()` alone, meant to be invoked once per call by something else (cron, a
systemd timer, a person typing `kanna scheduler tick`) — and, as it turned out, no way to actually
*create* a schedule from the CLI at all (`SchedulerStore.create()` existed but nothing called it
outside tests). This pass closes both gaps: `kanna scheduler add` to create schedules, and
`automation/scheduler/daemon.py::SchedulerDaemon` — a real "set and forget" foreground loop — as an
alternative to relying on an external scheduler to drive `tick`.

## Creating a schedule

```bash
kanna scheduler add "morning briefing" "How much did I spend on food this month?" \
    --kind weekly --weekday 0 --time 08:00 --anchor-date 2024-01-01

kanna scheduler add "reminder" "list the files in ./inbox" \
    --kind interval --seconds 3600

kanna scheduler add "one-off" "generate the quarterly report" \
    --kind once --run-at 2024-04-01T09:00:00
```

Every schedule's `request` is a plain natural-language string, run through the exact same
`kanna.agent_loop().run()` an interactive `kanna ask` would use — a scheduled job gets the same
planning, verification, and bounded correction as anything else, and if it needs a REVIEW-level
action, the same `TrustStoreGate` (see `docs/SECURITY.md`) decides whether it can run unattended. This
is exactly why trusted-automation configuration landed before this: a schedule that needs approval on
every fire isn't "set and forget" at all.

`kanna scheduler list` shows every schedule's id, active state, kind, and computed `next_run_at`.
`kanna scheduler remove <id>` deactivates one — a soft removal (`SchedulerStore.set_active(id, False)`)
rather than a hard delete, so its `jobs` history stays inspectable, consistent with "every run is
inspectable" (`KANNA_SPEC.md`'s design principles) applying here too.

## Two ways to run it: `tick` (still works) and `daemon` (new)

- **`kanna scheduler tick`** — runs every currently-due schedule once and exits. Unchanged from Phase
  1; still the right choice if you already have cron or a systemd timer driving it (`* * * * *
  kanna scheduler tick`, or similar).
- **`kanna scheduler daemon [--interval-seconds N] [--ticks N]`** — a real foreground process:
  `SchedulerDaemon.run_forever()` calls `tick()`, sleeps `interval_seconds` (default 60), and repeats
  until it receives SIGINT or SIGTERM (`install_signal_handlers()` wires both to a clean `stop()` —
  the loop finishes its current tick and exits, rather than being killed mid-job). `--ticks N` runs a
  bounded number of ticks and exits instead of running forever — mainly for manual verification or
  scripting, and how the CLI itself is tested (`tests/test_cli.py::test_scheduler_daemon_bounded_ticks`)
  without an actual wall-clock wait.

`SchedulerDaemon` takes its `sleep` function as a constructor argument (default `time.sleep`) for
exactly the reason `schedule.py`'s pure functions take an explicit `now` — so `tests/
test_scheduler.py`'s daemon tests run in milliseconds, injecting a no-op or recording sleep function,
never actually waiting on wall-clock time.

### Running it as a systemd service

For an actual "leave it running" deployment, a user-level systemd service is simpler than a login
shell staying open:

```ini
# ~/.config/systemd/user/kanna-scheduler.service
[Unit]
Description=Kanna scheduler daemon

[Service]
ExecStart=/usr/bin/env python3 /path/to/kanna-pa/main.py scheduler daemon --interval-seconds 60
Restart=on-failure
Environment=KANNA_HOME=%h/.kanna

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now kanna-scheduler.service
journalctl --user -u kanna-scheduler -f   # tail its output
```

This file isn't installed by anything — it's a documented starting point a user adapts (path,
interval, environment) for their own machine, the same spirit as `docs/DEVICES.md`'s manual Xvfb setup
instructions being something to copy, not something Kanna installs itself.

## Testing

`tests/test_scheduler.py` covers the pure `Schedule`/`compute_next_run`/`is_due` functions (every
schedule kind, including the spec's own "every two weeks on Tuesday" example), `Scheduler.tick()`
(due schedules actually run, a schedule with no executor is recorded as skipped rather than silently
dropped, an executor's exception is caught and recorded as a failed job rather than crashing the
tick), and `SchedulerDaemon` (bounded `run_n_ticks()` sleeps between ticks but not after the last,
`run_forever()` actually stops when `stop()` is called mid-loop, `install_signal_handlers()` wires
SIGINT to a real `stop()` call — verified by invoking the installed handler directly rather than
sending a real signal to the test process). `tests/test_cli.py` covers the full `add`/`list`/`tick`/
`remove`/`daemon --ticks` lifecycle through real subprocess invocations, plus `add` rejecting a
missing required field (`--run-at` for `--kind once`, etc.) with a clear error rather than crashing or
silently storing an unusable schedule.

## Known limitations

- One process per daemon instance, no distributed/multi-worker coordination — fine for a personal
  agent, not built for concurrent workers racing to claim the same due schedule.
- No systemd unit is actually installed by any Kanna command — the one above is documentation, not
  automation, mirroring `docs/DEVICES.md`'s stance on the manual Xvfb setup it documents.
- `kanna scheduler add` validates required fields per schedule kind, but not that a `once` schedule's
  `--run-at` (or a `weekly` schedule's `--anchor-date`) is a syntactically valid ISO date/datetime — a
  malformed one fails only later, inside `Schedule.__post_init__`/`compute_next_run()`, as a Python
  exception rather than a clean CLI error.
- No pause/resume distinction — `remove` (deactivate) is the only lifecycle transition besides
  creation; there's no "temporarily disable, then re-enable later" command (a removed schedule can't
  currently be reactivated except by direct database access).
