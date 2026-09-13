import sys
from types import SimpleNamespace

import pytest

from core.config.settings import Settings
from core.errors import LLMUnavailable
from core.llm.factory import build_provider
from core.llm.litellm_provider import LiteLLMProvider


def response():
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
        message=SimpleNamespace(content="answer", tool_calls=None))])


def test_ollama_fallbacks_have_per_call_endpoints(monkeypatch):
    calls = []
    def complete(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise RuntimeError("unavailable")
        return response()
    client = SimpleNamespace(completion=complete)
    monkeypatch.setitem(sys.modules, "litellm", client)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/")
    provider = LiteLLMProvider(model="ollama/first", fallback_models=["ollama/second", "gpt-4o-mini"])
    assert provider.complete([]).content == "answer"
    assert calls[0]["model"] == "openai/first"
    assert calls[1]["model"] == "openai/second"
    assert calls[1]["api_base"] == "http://localhost:11434/v1"
    assert "api_base" not in calls[2] and "api_key" not in calls[2]
    assert not hasattr(client, "api_base")
    assert provider.model == "ollama/first"


def test_provider_selection_respects_basic_mode():
    with pytest.raises(LLMUnavailable):
        build_provider(Settings(llm_provider="rule_based"))


def test_authoring_uses_configured_provider(ctx, monkeypatch):
    import json
    from core.llm.fake import FakeProvider
    from core.llm.base import LLMResponse
    from tools.authoring.tools import AuthorContentTool
    provider = FakeProvider([LLMResponse(content=json.dumps({"title": "Summary", "sections": [
        {"heading": "Finding", "paragraphs": ["Observed source"]}], "unresolved": []}))])
    seen = []
    monkeypatch.setattr("tools.authoring.tools.build_provider", lambda settings: (seen.append(settings), provider)[1])
    result = AuthorContentTool().execute({"source_text": "Observed source", "task": "Summarize"}, ctx)
    assert result.success
    assert seen == [ctx.settings]
    assert provider.calls


def test_assignment_uses_configured_provider(ctx, monkeypatch):
    from core.llm.fake import FakeProvider
    from core.llm.base import LLMResponse
    from tools.authoring.tools import AssignmentSolveTool
    provider = FakeProvider([LLMResponse(content='{"title":"Task","questions":[]}')])
    monkeypatch.setattr("tools.authoring.tools.build_provider", lambda settings: provider)
    result = AssignmentSolveTool().execute({"source_text": "Source", "task": "Solve"}, ctx)
    assert result.success
    assert provider.calls
