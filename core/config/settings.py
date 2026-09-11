"""Configuration system.

Layered resolution, lowest to highest priority:

    built-in defaults  →  config.toml (KANNA_CONFIG_PATH or ~/.kanna/config.toml)  →  KANNA_* env vars

Only the stdlib is used (`tomllib`, read-only — Kanna never needs to
*write* TOML, so no third-party writer is required).
"""
from __future__ import annotations

import os
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
    "llm_max_tokens": 4096,
    "llm_fallback_models": "ollama/llama3.2:3b,gpt-4o-mini",
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
    "log_level": "LOG_LEVEL",
    "max_plan_steps": "MAX_PLAN_STEPS",
    "max_corrections": "MAX_CORRECTIONS",
}

_INT_KEYS = {"llm_max_tokens", "max_plan_steps", "max_corrections"}


@dataclass(frozen=True)
class Settings:
    timezone: str = _DEFAULTS["timezone"]
    default_currency: str = _DEFAULTS["default_currency"]
    llm_provider: str = _DEFAULTS["llm_provider"]
    llm_model: str = _DEFAULTS["llm_model"]
    llm_max_tokens: int = _DEFAULTS["llm_max_tokens"]
    llm_fallback_models: str = _DEFAULTS["llm_fallback_models"]
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
        log_level=values["log_level"],
        max_plan_steps=int(values["max_plan_steps"]),
        max_corrections=int(values["max_corrections"]),
        sandbox_roots=tuple(values["sandbox_roots"]),
    )
