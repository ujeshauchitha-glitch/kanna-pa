"""A universal LLM provider backed by LiteLLM.

Supports 100+ providers (OpenAI, Anthropic, Google, local, free tiers)
through a single interface with automatic fallback when a provider hits
its token limit.  Set ``KANNA_LLM_MODEL`` to any LiteLLM-supported
model string (e.g. ``anthropic/claude-sonnet-5``, ``openai/gpt-4o``,
``gemini/gemini-2.0-flash``, ``ollama/llama3``).

Requires: ``pip install kanna[llm]`` (pulls in litellm).
"""
from __future__ import annotations

import os

from core.errors import LLMUnavailable
from core.llm.base import LLMResponse, Message, ToolCall


class LiteLLMProvider:
    def __init__(
        self,
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        fallback_models: list[str] | None = None,
    ) -> None:
        self.model = model or os.environ.get("KANNA_LLM_MODEL", "openai/gpt-4o-mini")
        self.max_tokens = max_tokens
        self.fallback_models = fallback_models or []
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import litellm
        except ImportError as exc:
            raise LLMUnavailable(
                "the 'litellm' package is not installed; run `pip install kanna[llm]`"
            ) from exc

        # Allow any provider key via standard env vars (OPENAI_API_KEY,
        # ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.) — litellm picks them
        # up automatically.
        self._client = litellm
        return self._client

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        litellm = self._get_client()

        litellm_messages = []
        if system:
            litellm_messages.append({"role": "system", "content": system})
        for m in messages:
            litellm_messages.append({"role": m.role, "content": m.content})

        kwargs: dict = {
            "model": self.model,
            "messages": litellm_messages,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                    },
                }
                for t in tools
            ]

        # Try primary model, then fallbacks
        models_to_try = [self.model] + self.fallback_models
        last_error: Exception | None = None

        for model in models_to_try:
            try:
                kwargs["model"] = model
                response = litellm.completion(**kwargs)
                return self._parse_response(response)
            except Exception as exc:
                last_error = exc
                continue

        raise LLMUnavailable(
            f"All models failed. Last error: {last_error}"
        ) from last_error

    def _parse_response(self, response) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        choice = response.choices[0]
        message = choice.message

        if message.content:
            text_parts.append(message.content)

        if message.tool_calls:
            for tc in message.tool_calls:
                import json
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, input=args)
                )

        return LLMResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "stop",
            raw=response,
        )
