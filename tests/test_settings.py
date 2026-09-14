"""core.config.settings — focused on the new llm_timeout_seconds field
(the rest of the resolution layering is already exercised indirectly by
every other test that constructs Settings) and update_config_file, the
narrow scalar-line TOML writer the desktop connection form uses."""
from __future__ import annotations

from core.config.settings import Settings, load_settings, update_config_file


def test_llm_timeout_seconds_has_a_sane_default():
    assert Settings().llm_timeout_seconds == 120


def test_llm_timeout_seconds_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("KANNA_HOME", str(tmp_path))
    monkeypatch.setenv("KANNA_LLM_TIMEOUT_SECONDS", "45")
    settings = load_settings(tmp_path / "no-such-config.toml")
    assert settings.llm_timeout_seconds == 45


# --- update_config_file ------------------------------------------------


def test_update_config_file_creates_a_new_file(tmp_path):
    path = tmp_path / "config.toml"
    update_config_file(path, {"llm_provider": "anthropic", "llm_timeout_seconds": 90})
    text = path.read_text()
    assert 'llm_provider = "anthropic"' in text
    assert "llm_timeout_seconds = 90" in text


def test_update_config_file_preserves_untouched_lines_exactly(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "# a comment kept verbatim\n"
        'timezone = "UTC"\n'
        "\n"
        'llm_provider = "litellm"\n'
        'llm_model = "ollama/qwen3:8b"\n'
    )
    update_config_file(path, {"llm_model": "claude-sonnet-5"})
    text = path.read_text()
    assert "# a comment kept verbatim\n" in text
    assert 'timezone = "UTC"\n' in text
    assert 'llm_provider = "litellm"\n' in text  # untouched key, untouched value
    assert 'llm_model = "claude-sonnet-5"' in text  # the one key actually changed
    assert "ollama/qwen3:8b" not in text


def test_update_config_file_appends_a_key_not_already_present(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('timezone = "UTC"\n')
    update_config_file(path, {"llm_provider": "rule_based"})
    text = path.read_text()
    assert 'timezone = "UTC"\n' in text
    assert 'llm_provider = "rule_based"' in text


def test_update_config_file_escapes_embedded_quotes_and_backslashes(tmp_path):
    path = tmp_path / "config.toml"
    update_config_file(path, {"llm_model": 'weird"model\\name'})
    # What actually matters: tomllib can parse it back to the exact value.
    import tomllib
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    assert data["llm_model"] == 'weird"model\\name'


def test_update_config_file_leaves_no_stray_tmp_file(tmp_path):
    path = tmp_path / "config.toml"
    update_config_file(path, {"llm_provider": "anthropic"})
    assert not (tmp_path / "config.toml.tmp").exists()


def test_update_config_file_round_trips_through_load_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("KANNA_HOME", str(tmp_path))
    monkeypatch.delenv("KANNA_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("KANNA_LLM_MODEL", raising=False)
    path = tmp_path / "config.toml"
    update_config_file(path, {
        "llm_provider": "anthropic", "llm_model": "claude-sonnet-5",
        "llm_fallback_models": "", "llm_timeout_seconds": 60,
    })
    settings = load_settings(path)
    assert settings.llm_provider == "anthropic"
    assert settings.llm_model == "claude-sonnet-5"
    assert settings.llm_timeout_seconds == 60
