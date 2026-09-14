"""Real socket-level checks for the single-instance guard — no mocking,
since the whole point is the actual OS-level bind/connect behavior."""
from __future__ import annotations

import threading
import time

from interfaces.desktop.singleton import SingleInstanceGuard, port_for


def test_port_for_is_deterministic_and_space_safe(tmp_path):
    home = tmp_path / "has spaces" / ".kanna"
    assert port_for(home) == port_for(home)
    assert port_for(home) != port_for(tmp_path / "other" / ".kanna")


def test_first_guard_acquires_second_does_not(tmp_path):
    home = tmp_path / ".kanna"
    first = SingleInstanceGuard(home)
    try:
        assert first.acquired
        second = SingleInstanceGuard(home)
        try:
            assert not second.acquired
        finally:
            second.close()
    finally:
        first.close()


def test_different_homes_both_acquire(tmp_path):
    a = SingleInstanceGuard(tmp_path / "a" / ".kanna")
    b = SingleInstanceGuard(tmp_path / "b" / ".kanna")
    try:
        assert a.acquired and b.acquired
    finally:
        a.close()
        b.close()


def test_notify_existing_reaches_running_instance(tmp_path):
    home = tmp_path / ".kanna"
    first = SingleInstanceGuard(home)
    shown = threading.Event()
    first.start(shown.set)
    try:
        second = SingleInstanceGuard(home)
        try:
            assert not second.acquired
            assert second.notify_existing() is True
            assert shown.wait(timeout=2.0)
        finally:
            second.close()
    finally:
        first.close()


def test_notify_existing_false_when_nothing_listening(tmp_path):
    # A port picked deterministically but never bound: nothing to notify.
    home = tmp_path / ".kanna"
    guard = SingleInstanceGuard(home)
    guard.close()
    time.sleep(0.05)
    assert guard.notify_existing(timeout=0.3) is False


def test_close_releases_the_port_for_reuse(tmp_path):
    home = tmp_path / ".kanna"
    first = SingleInstanceGuard(home)
    assert first.acquired
    first.close()
    second = SingleInstanceGuard(home)
    try:
        assert second.acquired
    finally:
        second.close()
