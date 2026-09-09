"""The `LLMProvider` protocol.

Anything that can turn a message list into a response — real or fake —
implements this. The rest of Kanna (planner, finance query phrasing,
etc.) only ever depends on this Protocol, never on a specific SDK, so
swapping providers or running fully offline (via `RuleBasedPlanner` +
`FakeProvider` in tests) never requires touching calling code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    raw: Any = None


class LLMProvider(Protocol):
    def complete(self, messages: list[Message], *, system: str | None = None,
                 tools: list[dict] | None = None) -> LLMResponse:
        ...
