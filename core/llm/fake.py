"""A scripted provider for tests.

Returns responses from a fixed queue (or a callable) instead of calling
any network. Nothing in `tests/` should ever import
`core.llm.anthropic_provider` — this is what stands in for it.
"""
from __future__ import annotations

from collections.abc import Callable

from core.llm.base import LLMResponse, Message


class FakeProvider:
    def __init__(self, responses: list[LLMResponse] | None = None,
                 responder: Callable[[list[Message]], LLMResponse] | None = None) -> None:
        self._responses = list(responses or [])
        self._responder = responder
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message], *, system: str | None = None,
                 tools: list[dict] | None = None) -> LLMResponse:
        self.calls.append(messages)
        if self._responder is not None:
            return self._responder(messages)
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="")
