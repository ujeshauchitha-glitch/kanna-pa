"""Assignment solver — extracts questions from source material and answers them.

Takes an assignment document (text) and a task description, uses the LLM
to identify individual questions/tasks, then answers each one using the
source material.  Returns structured content with per-question answers
and explicit unresolved items.

This is the first real assignment-solving workflow in Kanna.  It builds
on `author_content` but adds question extraction and per-question
answering rather than just producing a single block of authored content.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from core.llm.base import LLMProvider, Message

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Question:
    """One extracted assignment question."""
    id: str
    text: str


@dataclass
class Answer:
    """An answer to one question."""
    question_id: str
    question_text: str
    answer_text: str
    source_references: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)


@dataclass
class AssignmentResult:
    """Full assignment solving result."""
    title: str
    questions: list[Question] = field(default_factory=list)
    answers: list[Answer] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)


_EXTRACT_SYSTEM = """You are Kanna's assignment parser. Given source material and a task \
description, extract the individual questions or tasks that need to be answered.

Respond with ONLY a JSON object (no prose, no markdown fences) of the form:

{{
  "title": "<assignment title>",
  "questions": [
    {{"id": "Q1", "text": "<question text>"}},
    {{"id": "Q2", "text": "<question text>"}}
  ]
}}

Rules:
- Extract every distinct question or task from the source.
- Preserve the original question wording.
- If the source has numbered or lettered items, use those as ids.
- If no explicit questions exist, create logical task items.
- "title" should describe the assignment overall.

Source material:
{source}

Task: {task}"""

_ANSWER_SYSTEM = """You are Kanna's assignment solver. Given source material and a list of \
questions, answer each question using ONLY information from the source.

Respond with ONLY a JSON object (no prose, no markdown fences) of the form:

{{
  "answers": [
    {{
      "question_id": "Q1",
      "answer_text": "<detailed answer>",
      "source_references": ["<specific quotes or sections from the source used>"],
      "unresolved": ["<parts you could not answer from the source>"]
    }}
  ]
}}

Rules:
- Base ALL answers on the source material. Do not invent facts.
- Use specific references from the source to support each answer.
- If a question cannot be fully answered from the source, say so in "unresolved".
- Be thorough — provide complete answers, not just one-line summaries.
- Preserve any formulas, data, or technical details from the source.

Source material:
{source}

Questions:
{questions}"""


def _parse_json_response(content: str) -> dict:
    match = _JSON_BLOCK_RE.search(content)
    if not match:
        raise ValueError(f"LLM did not return JSON: {content[:200]!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}") from exc


def extract_questions(provider: LLMProvider, source_text: str, task: str) -> tuple[str, list[Question]]:
    """Use the LLM to extract individual questions from source material."""
    system = _EXTRACT_SYSTEM.format(source=source_text[:8000], task=task)
    response = provider.complete([Message(role="user", content=task)], system=system)
    payload = _parse_json_response(response.content)

    title = payload.get("title", "Assignment")
    questions = []
    for raw_q in payload.get("questions", []):
        if isinstance(raw_q, dict) and raw_q.get("text"):
            questions.append(Question(
                id=str(raw_q.get("id", f"Q{len(questions)+1}")),
                text=str(raw_q["text"]),
            ))
    return title, questions


def answer_questions(provider: LLMProvider, source_text: str,
                     questions: list[Question]) -> list[Answer]:
    """Use the LLM to answer each question using the source material."""
    if not questions:
        return []

    questions_text = "\n".join(f"- {q.id}: {q.text}" for q in questions)
    system = _ANSWER_SYSTEM.format(source=source_text[:8000], questions=questions_text)
    response = provider.complete(
        [Message(role="user", content=f"Answer these questions using the source material.")],
        system=system,
    )
    payload = _parse_json_response(response.content)

    answers = []
    for raw_a in payload.get("answers", []):
        if not isinstance(raw_a, dict):
            continue
        qid = str(raw_a.get("question_id", ""))
        # Find the matching question.
        q_text = next((q.text for q in questions if q.id == qid), "")
        answers.append(Answer(
            question_id=qid,
            question_text=q_text,
            answer_text=str(raw_a.get("answer_text", "")),
            source_references=[str(r) for r in raw_a.get("source_references", [])],
            unresolved=[str(u) for u in raw_a.get("unresolved", [])],
        ))

    return answers


def solve_assignment(provider: LLMProvider, source_text: str,
                     task: str) -> AssignmentResult:
    """Full assignment solving pipeline: extract questions, answer them."""
    title, questions = extract_questions(provider, source_text, task)

    # Skip answering if no questions were extracted.
    if not questions:
        provenance = [
            f"source: {len(source_text)} characters",
            f"task: {task}",
            f"questions extracted: 0",
            f"questions answered: 0",
        ]
        return AssignmentResult(
            title=title,
            questions=[],
            answers=[],
            unresolved=["No questions could be extracted from the source material."],
            provenance=provenance,
        )

    answers = answer_questions(provider, source_text, questions)

    # Collect all unresolved items.
    all_unresolved = []
    for answer in answers:
        all_unresolved.extend(answer.unresolved)

    provenance = [
        f"source: {len(source_text)} characters",
        f"task: {task}",
        f"questions extracted: {len(questions)}",
        f"questions answered: {len(answers)}",
    ]

    return AssignmentResult(
        title=title,
        questions=questions,
        answers=answers,
        unresolved=all_unresolved,
        provenance=provenance,
    )
