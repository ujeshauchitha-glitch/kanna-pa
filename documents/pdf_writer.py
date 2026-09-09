"""Render a `Document` to a .pdf via `reportlab`'s platypus flowables.

Generates the PDF directly (no LibreOffice/soffice dependency), so it
works the same way regardless of what else is installed on the host —
consistent with keeping every Phase 2 capability's real path as simple
and portable as possible.
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from core.errors import DocumentGenerationUnavailable
from documents.model import Document, TableData

_MAX_HEADING_LEVEL = 6


def render_pdf(document: Document, path: str | Path) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise DocumentGenerationUnavailable(
            "the 'reportlab' package is not installed; run `pip install kanna[documents]`"
        ) from exc

    styles = getSampleStyleSheet()
    story = [Paragraph(escape(document.title), styles["Title"])]
    if document.subtitle:
        story.append(Paragraph(escape(document.subtitle), styles["Heading2"]))
    if document.author:
        story.append(Paragraph(f"By {escape(document.author)}", styles["Normal"]))
    story.append(Spacer(1, 0.25 * inch))

    for section in document.sections:
        if section.heading:
            level = max(1, min(section.level, _MAX_HEADING_LEVEL))
            story.append(Paragraph(escape(section.heading), styles[f"Heading{level}"]))
        for paragraph in section.paragraphs:
            story.append(Paragraph(escape(paragraph), styles["Normal"]))
            story.append(Spacer(1, 0.08 * inch))
        for bullet in section.bullets:
            story.append(Paragraph(f"&bull;&nbsp;&nbsp;{escape(bullet)}", styles["Normal"]))
        if section.table is not None and section.table.headers:
            story.append(Spacer(1, 0.1 * inch))
            story.append(_build_table(section.table, Table, TableStyle, colors))
        story.append(Spacer(1, 0.2 * inch))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    doc.build(story)


def _build_table(table_data: TableData, Table, TableStyle, colors):  # noqa: N803 - reportlab classes
    data = [table_data.headers] + [[str(v) for v in row] for row in table_data.rows]
    table = Table(data)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    return table
