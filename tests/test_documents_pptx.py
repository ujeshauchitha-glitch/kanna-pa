from __future__ import annotations

import pytest

pptx = pytest.importorskip("pptx", reason="python-pptx not installed (kanna[documents] extra)")

from documents.model import Document, Section, TableData  # noqa: E402
from documents.pptx_writer import render_pptx  # noqa: E402


def test_render_pptx_one_slide_per_section_plus_title_slide(tmp_path):
    doc = Document(
        title="Deck", subtitle="Sub",
        sections=[
            Section(heading="Intro", bullets=["Point one", "Point two"]),
            Section(heading="Data only", table=TableData(headers=["A", "B"], rows=[["1", "2"]])),
        ],
    )
    path = tmp_path / "deck.pptx"
    render_pptx(doc, path)

    assert path.exists() and path.stat().st_size > 0

    prs = pptx.Presentation(str(path))
    slides = list(prs.slides)
    assert len(slides) == 3  # title slide + 2 sections

    assert slides[0].shapes.title.text == "Deck"
    assert slides[1].shapes.title.text == "Intro"
    # a table-only section must still carry its heading (regression test —
    # this previously landed on a titleless Blank layout).
    assert slides[2].shapes.title.text == "Data only"

    body_text = slides[1].placeholders[1].text_frame.text
    assert "Point one" in body_text


def test_render_pptx_table_only_section_includes_table(tmp_path):
    doc = Document(title="Deck", sections=[
        Section(heading="Numbers", table=TableData(headers=["X"], rows=[["1"], ["2"]])),
    ])
    path = tmp_path / "deck.pptx"
    render_pptx(doc, path)

    prs = pptx.Presentation(str(path))
    data_slide = list(prs.slides)[1]
    table_shapes = [s for s in data_slide.shapes if s.has_table]
    assert len(table_shapes) == 1
    table = table_shapes[0].table
    assert table.cell(0, 0).text == "X"
    assert table.cell(1, 0).text == "1"


def test_render_pptx_prefers_bullets_over_paragraphs(tmp_path):
    doc = Document(title="Deck", sections=[
        Section(heading="S", paragraphs=["ignored if bullets present"], bullets=["used"]),
    ])
    path = tmp_path / "deck.pptx"
    render_pptx(doc, path)

    prs = pptx.Presentation(str(path))
    body_text = list(prs.slides)[1].placeholders[1].text_frame.text
    assert body_text == "used"
