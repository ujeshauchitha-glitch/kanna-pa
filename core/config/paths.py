"""Filesystem locations Kanna uses for its own state.

All paths are derived from a single `KANNA_HOME` so tests and multiple
users on one machine can point Kanna at an isolated directory just by
setting an environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path


def kanna_home() -> Path:
    """Root directory for Kanna's persistent state.

    Resolution order: `KANNA_HOME` env var, else `~/.kanna`.
    """
    override = os.environ.get("KANNA_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".kanna"


def data_dir() -> Path:
    return kanna_home() / "data"


def db_path() -> Path:
    override = os.environ.get("KANNA_DB_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return data_dir() / "kanna.db"


def config_path() -> Path:
    override = os.environ.get("KANNA_CONFIG_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return kanna_home() / "config.toml"


def log_dir() -> Path:
    return kanna_home() / "logs"


def ensure_dirs() -> None:
    """Create every directory Kanna needs, if missing. Idempotent."""
    for d in (kanna_home(), data_dir(), log_dir()):
        d.mkdir(parents=True, exist_ok=True)
