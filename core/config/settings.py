"""Configuration system.

Layered resolution, lowest to highest priority:

    built-in defaults  →  config.toml (KANNA_CONFIG_PATH or ~/.kanna/config.toml)  →  KANNA_* env vars

Only the stdlib is used (`tomllib`, read-only for the *layered-resolution*
reader above). `update_config_file()` is a separate, deliberately narrow
writer for the one thing the desktop UI's connection settings form needs:
setting specific top-level scalar keys without disturbing anything else
already in the file — not a general TOML writer, so still no third-party
dependency.
"""
from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from core.config import paths

_DEFAULTS: dict[str, Any] = {
    "timezone": "UTC",
    "default_currency": "INR",
    "llm_provider": "litellm",
    "llm_model": "ollama/qwen3:8b",
    "llm_max_tokens": 8192,
    "llm_fallback_models": "ollama/llama3.2:3b,gpt-4o-mini",
    # A provider call has no way to be interrupted once sent (see
    # core/llm/*_provider.py), so cooperative cancellation — checked only
    # between calls, never mid-call — depends on a call eventually
    # returning at all. This bounds how long a hung/unreachable model
    # server (a local Ollama that never responds, say) can block a step.
    "llm_timeout_seconds": 120,
    "log_level": "INFO",
    "max_plan_steps": 20,
    "max_corrections": 3,
    "sandbox_roots": [],  # populated with kanna_home() at load time if empty
}

# Every setting that may be overridden by an environment variable, mapped
# name -> env var suffix (prefixed with KANNA_).
_ENV_KEYS = {
    "timezone": "TIMEZONE",
    "default_currency": "DEFAULT_CURRENCY",
    "llm_provider": "LLM_PROVIDER",
    "llm_model": "LLM_MODEL",
    "llm_max_tokens": "LLM_MAX_TOKENS",
    "llm_fallback_models": "LLM_FALLBACK_MODELS",
    "llm_timeout_seconds": "LLM_TIMEOUT_SECONDS",
    "log_level": "LOG_LEVEL",
    "max_plan_steps": "MAX_PLAN_STEPS",
    "max_corrections": "MAX_CORRECTIONS",
}

_INT_KEYS = {"llm_max_tokens", "llm_timeout_seconds", "max_plan_steps", "max_corrections"}


@dataclass(frozen=True)
class Settings:
    timezone: str = _DEFAULTS["timezone"]
    default_currency: str = _DEFAULTS["default_currency"]
    llm_provider: str = _DEFAULTS["llm_provider"]
    llm_model: str = _DEFAULTS["llm_model"]
    llm_max_tokens: int = _DEFAULTS["llm_max_tokens"]
    llm_fallback_models: str = _DEFAULTS["llm_fallback_models"]
    llm_timeout_seconds: int = _DEFAULTS["llm_timeout_seconds"]
    log_level: str = _DEFAULTS["log_level"]
    max_plan_steps: int = _DEFAULTS["max_plan_steps"]
    max_corrections: int = _DEFAULTS["max_corrections"]
    sandbox_roots: tuple[str, ...] = field(default_factory=tuple)

    def with_overrides(self, **kwargs: Any) -> "Settings":
        return Settings(**{**self.as_dict(), **kwargs})

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


_TOML_KEY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _toml_scalar_line(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"{key} = {'true' if value else 'false'}"
    if isinstance(value, (int, float)):
        return f"{key} = {value}"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'{key} = "{escaped}"'


def update_config_file(path: Path, updates: dict[str, Any]) -> None:
    """Set specific top-level scalar keys in `path`'s TOML, in place.

    Every other line — other keys, comments, blank lines, `[tables]` —
    is preserved exactly as written. A key already present is replaced
    on its existing line; a new key is appended. This intentionally
    does not parse or write tables/arrays/multi-line values: every
    setting `load_settings()` reads is a plain top-level scalar (see
    `_DEFAULTS` above), and that's the only thing this ever needs to
    write. The write is atomic (write to a temp file, then replace) so
    a crash mid-write can't leave a half-written config behind.
    """
    path = Path(path)
    lines = path.read_text().splitlines() if path.exists() else []
    remaining = dict(updates)
    out: list[str] = []
    for line in lines:
        match = _TOML_KEY_RE.match(line)
        if match and match.group(1) in remaining:
            out.append(_toml_scalar_line(match.group(1), remaining.pop(match.group(1))))
        else:
            out.append(line)
    for key, value in remaining.items():
        out.append(_toml_scalar_line(key, value))

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(out) + ("\n" if out else ""))
    tmp.replace(path)


def load_settings(config_path: Path | None = None) -> Settings:
    """Resolve settings from defaults -> config.toml -> environment."""
    values = dict(_DEFAULTS)

    toml_path = config_path or paths.config_path()
    file_values = _load_toml(toml_path)
    for key in _DEFAULTS:
        if key in file_values:
            values[key] = file_values[key]

    for key, suffix in _ENV_KEYS.items():
        env_val = os.environ.get(f"KANNA_{suffix}")
        if env_val is None:
            continue
        values[key] = int(env_val) if key in _INT_KEYS else env_val

    if not values["sandbox_roots"]:
        # Default to the current working directory (so "list the files in
        # ./docs" works out of the box for whatever project the user is
        # in) plus Kanna's own home (where it keeps its DB/config, and a
        # safe place to write to regardless of cwd).
        values["sandbox_roots"] = [str(Path.cwd()), str(paths.kanna_home())]

    return Settings(
        timezone=values["timezone"],
        default_currency=values["default_currency"],
        llm_provider=values["llm_provider"],
        llm_model=values["llm_model"],
        llm_max_tokens=int(values["llm_max_tokens"]),
        llm_fallback_models=values.get("llm_fallback_models", ""),
        llm_timeout_seconds=int(values.get("llm_timeout_seconds", _DEFAULTS["llm_timeout_seconds"])),
        log_level=values["log_level"],
        max_plan_steps=int(values["max_plan_steps"]),
        max_corrections=int(values["max_corrections"]),
        sandbox_roots=tuple(values["sandbox_roots"]),
    )
