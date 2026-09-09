"""The document content model every writer renders from.

One structured representation — `Document`/`Section`/`TableData` — feeds
all three writers (`docx_writer.py`, `pptx_writer.py`, `pdf_writer.py`),
the same way `finance.money.Money`/`finance.models.Transaction` feed
every finance entry path. Callers (tools, and eventually higher-level
report/assignment workflows) build one `Document` and can render it to
any format without re-describing the content three times.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TableData:
    headers: list[str]
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class Section:
    heading: str | None = None
    level: int = 1  # 1 = top-level (H1 / its own slide), 2 = subsection, ...
    paragraphs: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    table: TableData | None = None


@dataclass
class Document:
    title: str
    subtitle: str | None = None
    author: str | None = None
    sections: list[Section] = field(default_factory=list)


def build_document(args: dict) -> Document:
    """Convert a tool's raw JSON-shaped args into a `Document`.

    Shared by every `documents.tools` generator so the "args dict ->
    Document" parsing exists exactly once. `args["sections"]` is expected
    to already have passed `SECTIONS_SCHEMA` validation (see
    `documents/tools.py`), so this only does structural conversion, not
    re-validation.
    """
    sections = []
    for raw in args.get("sections", []):
        table = None
        raw_table = raw.get("table")
        if raw_table:
            table = TableData(headers=list(raw_table.get("headers", [])),
                               rows=[list(row) for row in raw_table.get("rows", [])])
        sections.append(Section(
            heading=raw.get("heading") or None,
            level=int(raw.get("level") or 1),
            paragraphs=list(raw.get("paragraphs", [])),
            bullets=list(raw.get("bullets", [])),
            table=table,
        ))

    return Document(
        title=args["title"],
        subtitle=args.get("subtitle") or None,
        author=args.get("author") or None,
        sections=sections,
    )
