from __future__ import annotations

from documents.model import Section, TableData, build_document


def test_build_document_minimal():
    doc = build_document({"title": "Report", "path": "x.docx"})
    assert doc.title == "Report"
    assert doc.subtitle is None
    assert doc.author is None
    assert doc.sections == []


def test_build_document_full():
    args = {
        "title": "Report", "subtitle": "Subtitle", "author": "Author",
        "sections": [
            {"heading": "Intro", "level": 1, "paragraphs": ["hello"], "bullets": ["a", "b"]},
            {"heading": "Data", "table": {"headers": ["A", "B"], "rows": [["1", "2"]]}},
        ],
    }
    doc = build_document(args)
    assert doc.subtitle == "Subtitle"
    assert doc.author == "Author"
    assert len(doc.sections) == 2
    assert doc.sections[0] == Section(heading="Intro", level=1, paragraphs=["hello"], bullets=["a", "b"])
    assert doc.sections[1].table == TableData(headers=["A", "B"], rows=[["1", "2"]])


def test_build_document_empty_strings_become_none():
    doc = build_document({"title": "T", "subtitle": "", "author": "", "sections": []})
    assert doc.subtitle is None
    assert doc.author is None


def test_build_document_default_level_is_one():
    doc = build_document({"title": "T", "sections": [{"heading": "H"}]})
    assert doc.sections[0].level == 1
