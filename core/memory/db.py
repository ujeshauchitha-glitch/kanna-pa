"""SQLite-backed persistence layer.

`Database` owns the connection and migration application. Everything
else (repositories) talks to SQLite only through this object, so
pragmas, locking, and row-factory behavior are configured in exactly one
place.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_VERSION_RE = re.compile(r"^(\d+)_")


def _discover_migrations() -> list[tuple[int, Path]]:
    migrations = []
    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        match = _VERSION_RE.match(path.name)
        if not match:
            continue
        migrations.append((int(match.group(1)), path))
    return sorted(migrations, key=lambda m: m[0])


class Database:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._ensure_migrations_table()

    def _ensure_migrations_table(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        self._conn.commit()

    def migrate(self) -> list[int]:
        """Apply every migration not yet recorded. Idempotent — safe to call repeatedly."""
        from datetime import datetime, timezone

        applied: list[int] = []
        with self._lock:
            existing = {
                row["version"] for row in self._conn.execute("SELECT version FROM schema_migrations")
            }
            for version, path in _discover_migrations():
                if version in existing:
                    continue
                sql = path.read_text()
                self._conn.executescript(sql)
                self._conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (version, datetime.now(timezone.utc).isoformat()),
                )
                self._conn.commit()
                applied.append(version)
        return applied

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params))

    def query_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cur.fetchone()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
