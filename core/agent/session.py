"""Conversation/session management.

A thin façade over `SessionRepository` that the CLI and agent loop share,
so starting a session, logging turns, and ending it is one call each
instead of every caller touching the repository directly.
"""
from __future__ import annotations

from core.memory.db import Database
from core.memory.repositories.sessions import Message, Session, SessionRepository


class AgentSession:
    def __init__(self, db: Database, metadata: dict | None = None) -> None:
        self._repo = SessionRepository(db)
        self._session = self._repo.create(metadata)

    @property
    def id(self) -> str:
        return self._session.id

    def log_user_message(self, content: str) -> Message:
        return self._repo.add_message(self.id, "user", content)

    def log_assistant_message(self, content: str) -> Message:
        return self._repo.add_message(self.id, "assistant", content)

    def history(self) -> list[Message]:
        return self._repo.get_messages(self.id)

    def end(self) -> None:
        self._repo.end(self.id)
