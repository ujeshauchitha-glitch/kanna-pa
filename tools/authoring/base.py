"""Content author protocol and result types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class AuthoredSection:
    """One authored section — maps directly to `documents.model.Section`."""
    heading: str | None = None
    level: int = 1
    paragraphs: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)


@dataclass
class AuthoredContent:
    """Structured content produced by an author.

    The `sections` list is directly compatible with the document
    generation tools' `sections` parameter.  `unresolved` carries
    explicit questions the author could not answer from the source
    alone — the caller must decide whether to ask the user or skip.
    `provenance` records which sources were actually read and used.
    """
    title: str
    sections: list[AuthoredSection] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)


class ContentAuthor(Protocol):
    """Protocol for content authoring implementations."""

    def author(self, *, source_text: str, task: str, title: str | None = None) -> AuthoredContent:
        """Author structured content from source material and a task description."""
        ...
