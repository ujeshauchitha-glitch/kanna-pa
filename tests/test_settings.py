"""core.config.settings — focused on the new llm_timeout_seconds field
(the rest of the resolution layering is already exercised indirectly by
every other test that constructs Settings)."""
from __future__ import annotations

from core.config.settings import Settings, load_settings


def test_llm_timeout_seconds_has_a_sane_default():
    assert Settings().llm_timeout_seconds == 120


def test_llm_timeout_seconds_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("KANNA_HOME", str(tmp_path))
    monkeypatch.setenv("KANNA_LLM_TIMEOUT_SECONDS", "45")
    settings = load_settings(tmp_path / "no-such-config.toml")
    assert settings.llm_timeout_seconds == 45
