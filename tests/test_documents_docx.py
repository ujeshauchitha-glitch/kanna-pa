from __future__ import annotations

import pytest

docx = pytest.importorskip("docx", reason="python-docx not installed (kanna[documents] extra)")

from documents.model import Document, Section, TableData  # noqa: E402
from documents.docx_writer import render_docx  # noqa: E402


def test_render_docx_creates_file_with_expected_structure(tmp_path):
    doc = Document(
        title="Report", subtitle="Sub", author="Ada",
        sections=[
            Section(heading="Intro", level=1, paragraphs=["Hello world."]),
            Section(heading="Findings", level=1, bullets=["First", "Second"]),
            Section(heading="Data", level=2,
                    table=TableData(headers=["A", "B"], rows=[["1", "2"], ["3", "4"]])),
        ],
    )
    path = tmp_path / "report.docx"
    render_docx(doc, path)

    assert path.exists()
    assert path.stat().st_size > 0

    result = docx.Document(str(path))
    texts = [p.text for p in result.paragraphs]
    assert "Report" in texts
    assert "Sub" in texts
    assert "By Ada" in texts
    assert "Intro" in texts
    assert "Hello world." in texts
    assert "First" in texts and "Second" in texts

    assert len(result.tables) == 1
    table = result.tables[0]
    assert len(table.rows) == 3  # header + 2 data rows
    assert table.rows[0].cells[0].text == "A"
    assert table.rows[1].cells[1].text == "2"


def test_render_docx_minimal_document(tmp_path):
    doc = Document(title="Just a title")
    path = tmp_path / "minimal.docx"
    render_docx(doc, path)
    assert path.exists() and path.stat().st_size > 0


def test_render_docx_creates_parent_directories(tmp_path):
    doc = Document(title="T")
    path = tmp_path / "nested" / "dir" / "report.docx"
    render_docx(doc, path)
    assert path.exists()
