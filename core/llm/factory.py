"""Shared provider selection for planning and source-aware authoring."""
from core.errors import LLMUnavailable


def build_provider(settings):
    if settings.llm_provider == "litellm":
        from core.llm.litellm_provider import LiteLLMProvider
        provider = LiteLLMProvider(model=settings.llm_model, max_tokens=settings.llm_max_tokens,
            fallback_models=[m.strip() for m in settings.llm_fallback_models.split(",") if m.strip()],
            timeout=settings.llm_timeout_seconds)
    elif settings.llm_provider == "anthropic":
        from core.llm.anthropic_provider import AnthropicProvider
        provider = AnthropicProvider(model=settings.llm_model, max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds)
    else:
        raise LLMUnavailable("No LLM configured. Select litellm or anthropic to author content.")
    provider._get_client()
    return provider
