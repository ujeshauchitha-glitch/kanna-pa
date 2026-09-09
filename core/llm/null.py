"""A provider that always reports unavailability. Used as the safe default
when no LLM is configured and no fallback should silently kick in."""
from __future__ import annotations

from core.errors import LLMUnavailable
from core.llm.base import LLMResponse, Message


class NullProvider:
    def complete(self, messages: list[Message], *, system: str | None = None,
                 tools: list[dict] | None = None) -> LLMResponse:
        raise LLMUnavailable("no LLM provider is configured")
