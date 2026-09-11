"""LLM-backed content author.

Uses the same `LLMProvider` protocol as the planner — a model reads the
source material and the task description, then returns structured
Document-compatible content.  The model is explicitly instructed to:

1. Base all factual claims on the provided source text.
2. Preserve tables and data from the source where relevant.
3. Flag questions it cannot answer rather than guessing.
4. Record which sources it used in provenance.

The output is schema-validated before returning, so a malformed model
response never produces invalid Document content downstream.
"""
from __future__ import annotations

import json
import re

from core.llm.base import LLMProvider, Message
from tools.authoring.base import AuthoredContent, AuthoredSection

_SYSTEM_TEMPLATE = """You are Kanna's content author. You receive source material and a task. \
Your job is to produce structured document content — title, sections with headings, paragraphs, \
and bullet points — that fulfills the task using only information present in the source.

Respond with ONLY a JSON object (no prose, no markdown fences) of the form:

{{
  "title": "<document title>",
  "sections": [
    {{
      "heading": "<section heading>",
      "level": 1,
      "paragraphs": ["<paragraph text>", ...],
      "bullets": ["<bullet point>", ...]
    }}
  ],
  "unresolved": ["<question you cannot answer from the source alone>", ...]
}}

Rules:
- Base ALL factual claims on the source text. Do not invent facts, statistics, or quotes.
- Preserve tables as text paragraphs if present in the source — describe their content.
- Use headings that match the task's structure (e.g. assignment question numbers).
- "unresolved" must list every question or requirement you could NOT fully address from the \
source alone. An empty list means you addressed everything.
- If the task is an assignment, answer each question directly using the source.
- If the task is a report, organize findings logically with clear section headings.
- If the source is insufficient, say so in "unresolved" rather than fabricating answers.

Source material:
{source}

Task: {task}

{title_line}"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(content: str, source_text: str, task: str, title: str | None) -> AuthoredContent:
    """Parse the LLM's JSON response into an AuthoredContent."""
    match = _JSON_BLOCK_RE.search(content)
    if not match:
        raise ValueError(f"LLM did not return JSON: {content[:200]!r}")

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("LLM response must be an object")

    result_title = payload.get("title") or title or "Untitled"
    sections = []
    for raw_section in payload.get("sections", []):
        if not isinstance(raw_section, dict):
            continue
        sections.append(AuthoredSection(
            heading=raw_section.get("heading") or None,
            level=int(raw_section.get("level") or 1),
            paragraphs=list(raw_section.get("paragraphs", [])),
            bullets=list(raw_section.get("bullets", [])),
        ))

    unresolved = [str(q) for q in payload.get("unresolved", []) if q]

    # Build provenance from what was actually provided.
    provenance = []
    if source_text.strip():
        provenance.append(f"source: {len(source_text)} characters of source material")
    provenance.append(f"task: {task}")
    if title:
        provenance.append(f"title: {title}")

    return AuthoredContent(
        title=result_title,
        sections=sections,
        unresolved=unresolved,
        provenance=provenance,
    )


class LLMAuthor:
    """Content author backed by an `LLMProvider`.

    Same provider abstraction as `LLMPlanner` — real or fake in tests.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def author(self, *, source_text: str, task: str, title: str | None = None) -> AuthoredContent:
        title_line = f"Document title: {title}" if title else "Choose an appropriate title."
        system = _SYSTEM_TEMPLATE.format(
            source=source_text[:8000],  # guard against context overflow
            task=task,
            title_line=title_line,
        )

        response = self.provider.complete(
            [Message(role="user", content=task)],
            system=system,
        )

        return _parse_response(response.content, source_text, task, title)
