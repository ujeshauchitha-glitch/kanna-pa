"""A minimal synchronous publish/subscribe event bus.

Deliberately simple: handlers run inline, in registration order, on the
publishing thread. That's enough for Phase 1 (persisting events, driving
CLI progress output) without pulling in async machinery the rest of the
core doesn't need yet.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from core.events.types import Event

Handler = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._wildcard_handlers: list[Handler] = []

    def subscribe(self, topic_prefix: str, handler: Handler) -> None:
        """Register `handler` for every event whose topic starts with `topic_prefix`.

        Pass an empty string to subscribe to everything.
        """
        if topic_prefix == "":
            self._wildcard_handlers.append(handler)
        else:
            self._handlers[topic_prefix].append(handler)

    def unsubscribe(self, topic_prefix: str, handler: Handler) -> None:
        if topic_prefix == "":
            if handler in self._wildcard_handlers:
                self._wildcard_handlers.remove(handler)
            return
        handlers = self._handlers.get(topic_prefix, [])
        if handler in handlers:
            handlers.remove(handler)

    def publish(self, event: Event) -> None:
        for handler in self._wildcard_handlers:
            handler(event)
        for prefix, handlers in self._handlers.items():
            if event.topic.startswith(prefix):
                for handler in handlers:
                    handler(event)
