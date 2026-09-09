"""The real LLM provider, backed by the Anthropic API.

Lazily imports `anthropic` so the rest of Kanna never needs it installed
(`pip install kanna[llm]` pulls it in). Raises `LLMUnavailable` — never
silently falls back — when the package or an API key is missing; callers
that want a fallback (the planner does) catch that explicitly and switch
to `RuleBasedPlanner`.
"""
from __future__ import annotations

import os

from core.errors import LLMUnavailable
from core.llm.base import LLMResponse, Message, ToolCall


class AnthropicProvider:
    def __init__(self, *, model: str = "claude-sonnet-5", max_tokens: int = 4096,
                 api_key: str | None = None) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self._api_key:
            raise LLMUnavailable(
                "ANTHROPIC_API_KEY is not set; either export it or configure a "
                "different llm_provider (e.g. leave the planner on rule_based)"
            )
        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailable(
                "the 'anthropic' package is not installed; run `pip install kanna[llm]`"
            ) from exc

        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, messages: list[Message], *, system: str | None = None,
                 tools: list[dict] | None = None) -> LLMResponse:
        client = self._get_client()

        anthropic_messages = [{"role": m.role, "content": m.content} for m in messages]
        anthropic_tools = None
        if tools:
            anthropic_tools = [
                {"name": t["name"], "description": t.get("description", ""),
                 "input_schema": t.get("input_schema", {"type": "object", "properties": {}})}
                for t in tools
            ]

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))

        return LLMResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason or "end_turn",
            raw=response,
        )
