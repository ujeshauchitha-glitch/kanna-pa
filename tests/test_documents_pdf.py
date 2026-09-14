from __future__ import annotations

import pytest

reportlab = pytest.importorskip("reportlab", reason="reportlab not installed (kanna[documents] extra)")

from documents.model import Document, Section, TableData  # noqa: E402
from documents.pdf_writer import render_pdf  # noqa: E402


def test_render_pdf_creates_valid_file(tmp_path):
    doc = Document(
        title="Report", subtitle="Sub", author="Ada",
        sections=[
            Section(heading="Intro", level=1, paragraphs=["Hello world."]),
            Section(heading="Findings", level=2, bullets=["First", "Second"]),
            Section(heading="Data", level=1,
                    table=TableData(headers=["A", "B"], rows=[["1", "2"], ["3", "4"]])),
        ],
    )
    path = tmp_path / "report.pdf"
    render_pdf(doc, path)

    assert path.exists()
    data = path.read_bytes()
    assert data[:5] == b"%PDF-"
    assert len(data) > 500  # a trivially-empty/broken PDF would be far smaller


def test_render_pdf_minimal_document(tmp_path):
    path = tmp_path / "minimal.pdf"
    render_pdf(Document(title="Just a title"), path)
    assert path.exists()
    assert path.read_bytes()[:5] == b"%PDF-"


def test_render_pdf_escapes_special_characters_without_raising(tmp_path):
    """Regression test: unescaped '&'/'<'/'>' would make reportlab's Paragraph
    parser raise (it treats content as a minimal XML-like markup language)."""
    doc = Document(
        title="Q&A <Report> for A&B Corp",
        sections=[Section(heading="Terms & Conditions",
                           paragraphs=["Use of < and > and & is common in real text."])],
    )
    path = tmp_path / "escaped.pdf"
    render_pdf(doc, path)  # must not raise
    assert path.exists() and path.stat().st_size > 0


def test_render_pdf_heading_level_clamped_beyond_six(tmp_path):
    doc = Document(title="T", sections=[Section(heading="Deep", level=99, paragraphs=["x"])])
    path = tmp_path / "deep.pdf"
    render_pdf(doc, path)  # must not raise on an out-of-range Heading style lookup
    assert path.exists()


def test_render_pdf_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "report.pdf"
    render_pdf(Document(title="T"), path)
    assert path.exists()
