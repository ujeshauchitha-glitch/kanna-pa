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
        timeout: float | None = 120.0,
    ) -> None:
        self.model = model or os.environ.get("KANNA_LLM_MODEL", "openai/gpt-4o-mini")
        self.max_tokens = max_tokens
        self.fallback_models = fallback_models or []
        # A request with no bound can hang indefinitely — cooperative
        # cancellation (core.agent.loop.AgentLoop.run) only checks
        # *between* calls, so an in-flight call still needs its own
        # bound (per attempt: primary + every fallback each get up to
        # this long) for cancellation to mean anything in practice.
        self.timeout = timeout
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
        # up automatically.  For Ollama, route through the OpenAI-compatible
        # /v1/ endpoint which handles Qwen's thinking tags properly.
        # Endpoint and credentials belong to each call, not module globals:
        # a cloud fallback must never inherit Ollama's local endpoint/key.

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
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
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
                call_kwargs = {**kwargs, "model": model}
                if model.startswith("ollama/"):
                    call_kwargs.update(model="openai/" + model.split("/", 1)[1],
                        api_base=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1",
                        api_key="ollama")
                response = litellm.completion(**call_kwargs)
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
            content = message.content
            # Qwen 3 wraps responses in <think>...</think> tags — strip them
            if "<think>" in content and "</think>" in content:
                import re
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if content:
                text_parts.append(content)

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
