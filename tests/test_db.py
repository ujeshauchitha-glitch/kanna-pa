from __future__ import annotations

import sqlite3

import pytest

from core.memory.db import Database


def test_migrate_creates_tables():
    db = Database(":memory:")
    applied = db.migrate()
    assert applied == [1, 2]
    tables = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "finance_transactions" in tables
    assert "sessions" in tables
    assert "plan_steps" in tables
    assert "trust_rules" in tables


def test_migrate_is_idempotent():
    db = Database(":memory:")
    db.migrate()
    second_run = db.migrate()
    assert second_run == []  # nothing new to apply


def test_foreign_keys_enforced():
    db = Database(":memory:")
    db.migrate()
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            ("nonexistent-session", "user", "hi", "2024-01-01T00:00:00Z"),
        )


def test_query_and_query_one():
    db = Database(":memory:")
    db.migrate()
    db.execute(
        "INSERT INTO sessions (id, started_at, metadata) VALUES (?, ?, ?)",
        ("s1", "2024-01-01T00:00:00Z", "{}"),
    )
    row = db.query_one("SELECT * FROM sessions WHERE id = ?", ("s1",))
    assert row["id"] == "s1"
    rows = db.query("SELECT * FROM sessions")
    assert len(rows) == 1


def test_file_backed_db_persists_across_connections(tmp_path):
    path = tmp_path / "kanna.db"
    db1 = Database(path)
    db1.migrate()
    db1.execute("INSERT INTO sessions (id, started_at, metadata) VALUES (?, ?, ?)",
                ("s1", "2024-01-01T00:00:00Z", "{}"))
    db1.close()

    db2 = Database(path)
    db2.migrate()  # idempotent even on an already-migrated file
    row = db2.query_one("SELECT * FROM sessions WHERE id = ?", ("s1",))
    assert row is not None
    db2.close()
