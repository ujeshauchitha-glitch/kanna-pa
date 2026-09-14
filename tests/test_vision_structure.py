from __future__ import annotations

from vision.document.anthropic_document import parse_structure_json
from vision.document.base import DocumentStructure, StructureSection, StructureTable
from vision.document.fake import FakeDocumentProvider


def test_fake_document_provider_structure_extraction():
    extraction = DocumentStructure(
        title="Assignment 3",
        sections=[StructureSection(heading="Question 1", paragraphs=["Explain X."])],
    )
    provider = FakeDocumentProvider(structure_extractions=[extraction])
    result = provider.extract_structure(b"bytes", mime_type="application/pdf")
    assert result.title == "Assignment 3"
    assert result.sections[0].heading == "Question 1"
    assert provider.structure_calls == [(b"bytes", "application/pdf")]


def test_fake_document_provider_structure_default_is_empty():
    provider = FakeDocumentProvider()
    result = provider.extract_structure(b"x", mime_type="application/pdf")
    assert result == DocumentStructure()


# -- parse_structure_json: pure JSON -> DocumentStructure conversion --

def test_parse_structure_json_full():
    data = {
        "title": "Quarterly Report",
        "sections": [
            {"heading": "Overview", "level": 1, "paragraphs": ["Revenue grew."], "bullets": [],
             "table": None},
            {"heading": "Figures", "level": 2, "paragraphs": [], "bullets": ["Point one"],
             "table": {"headers": ["Q1", "Q2"], "rows": [["100", "120"]]}},
        ],
        "notes": "",
    }
    result = parse_structure_json(data, raw_text="raw ocr text")
    assert result.title == "Quarterly Report"
    assert len(result.sections) == 2
    assert result.sections[0] == StructureSection(
        heading="Overview", level=1, paragraphs=["Revenue grew."], bullets=[], table=None)
    assert result.sections[1].table == StructureTable(headers=["Q1", "Q2"], rows=[["100", "120"]])
    assert result.raw_text == "raw ocr text"


def test_parse_structure_json_all_null():
    data = {"title": None, "sections": [], "notes": "document too blurry to read"}
    result = parse_structure_json(data)
    assert result.title is None
    assert result.sections == []
    assert result.notes == "document too blurry to read"


def test_parse_structure_json_tolerates_wrong_types():
    data = {"title": 123, "sections": "not a list"}
    result = parse_structure_json(data)
    assert result.title is None
    assert result.sections == []
    assert "title" in result.notes
    assert "sections" in result.notes


def test_parse_structure_json_skips_malformed_sections_keeps_valid_ones():
    data = {"sections": [
        {"heading": "ok section", "paragraphs": ["text"]},
        "junk",
        {"heading": "no level given"},
    ]}
    result = parse_structure_json(data)
    # "junk" is skipped with a note; the malformed-but-dict section still
    # produces a section (missing fields default to empty/level 1) rather
    # than being dropped, the same tolerance parse_statement_json applies.
    assert len(result.sections) == 2
    assert result.sections[0].heading == "ok section"
    assert result.sections[1].level == 1
    assert "section 1" in result.notes


def test_parse_structure_json_defaults_missing_level_to_one():
    data = {"sections": [{"heading": "no level"}]}
    result = parse_structure_json(data)
    assert result.sections[0].level == 1


def test_parse_structure_json_rejects_non_positive_level():
    data = {"sections": [{"heading": "bad level", "level": 0}]}
    result = parse_structure_json(data)
    assert result.sections[0].level == 1


def test_parse_structure_json_tolerates_malformed_table():
    data = {"sections": [{"heading": "h", "table": "not an object"}]}
    result = parse_structure_json(data)
    assert result.sections[0].table is None
    assert "table" in result.notes


def test_parse_structure_json_table_skips_malformed_rows_keeps_valid_ones():
    data = {"sections": [{"heading": "h", "table": {
        "headers": ["A", "B"],
        "rows": [["1", "2"], "not a row", ["3", "4"]],
    }}]}
    result = parse_structure_json(data)
    table = result.sections[0].table
    assert table.headers == ["A", "B"]
    assert table.rows == [["1", "2"], ["3", "4"]]
    assert "table row" in result.notes
