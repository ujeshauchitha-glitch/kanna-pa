"""Tests against the real `documents.convert.convert_to_pdf` — skipped if
no LibreOffice binary is on PATH. The "binary missing" path is tested
unconditionally (via monkeypatch), since it doesn't need LibreOffice
installed to prove. `tests/test_documents_tools.py` covers the
`document_convert_to_pdf` tool layer via monkeypatching `convert_to_pdf`
directly, so that coverage doesn't depend on LibreOffice either.
"""
from __future__ import annotations

import pytest

pytest.importorskip("docx", reason="python-docx not installed (kanna[documents] extra)")
pytest.importorskip("pptx", reason="python-pptx not installed (kanna[documents] extra)")

from core.errors import DocumentConversionUnavailable  # noqa: E402
from documents.convert import convert_to_pdf, is_available  # noqa: E402
from documents.docx_writer import render_docx  # noqa: E402
from documents.model import Document, Section  # noqa: E402
from documents.pptx_writer import render_pptx  # noqa: E402

_needs_soffice = pytest.mark.skipif(
    not is_available(), reason="no LibreOffice ('soffice'/'libreoffice') binary on PATH"
)

_DOC = Document(title="Report", sections=[Section(heading="Intro", paragraphs=["Hello, world."])])


@_needs_soffice
def test_convert_docx_to_pdf(tmp_path):
    source = tmp_path / "in.docx"
    render_docx(_DOC, source)
    dest = tmp_path / "out.pdf"

    convert_to_pdf(source, dest)

    assert dest.exists()
    assert dest.read_bytes()[:5] == b"%PDF-"
    assert dest.stat().st_size > 100


@_needs_soffice
def test_convert_pptx_to_pdf(tmp_path):
    source = tmp_path / "in.pptx"
    render_pptx(_DOC, source)
    dest = tmp_path / "out.pdf"

    convert_to_pdf(source, dest)

    assert dest.exists()
    assert dest.read_bytes()[:5] == b"%PDF-"


@_needs_soffice
def test_convert_writes_exactly_to_dest_path_not_soffices_own_naming(tmp_path):
    """soffice always names its own output <source-stem>.pdf — proves the
    move-into-place logic actually lands the file at the caller's exact
    `dest`, not wherever soffice happened to put it."""
    source = tmp_path / "some_report.docx"
    render_docx(_DOC, source)
    dest = tmp_path / "totally_different_name.pdf"

    convert_to_pdf(source, dest)

    assert dest.exists()
    assert not (tmp_path / "some_report.pdf").exists()


@_needs_soffice
def test_convert_creates_dest_parent_directories(tmp_path):
    source = tmp_path / "in.docx"
    render_docx(_DOC, source)
    dest = tmp_path / "nested" / "dir" / "out.pdf"

    convert_to_pdf(source, dest)

    assert dest.exists()


@_needs_soffice
def test_convert_missing_source_raises(tmp_path):
    with pytest.raises(DocumentConversionUnavailable, match="does not exist"):
        convert_to_pdf(tmp_path / "nope.docx", tmp_path / "out.pdf")


@_needs_soffice
def test_convert_timeout_raises(tmp_path):
    source = tmp_path / "in.docx"
    render_docx(_DOC, source)

    with pytest.raises(DocumentConversionUnavailable, match="timed out"):
        convert_to_pdf(source, tmp_path / "out.pdf", timeout=0.001)


def test_is_available_reflects_the_real_binary_presence():
    import shutil
    expected = shutil.which("soffice") is not None or shutil.which("libreoffice") is not None
    assert is_available() == expected


def test_no_binary_raises_document_conversion_unavailable(tmp_path, monkeypatch):
    # Doesn't need LibreOffice installed at all — proves the honest
    # "capability unavailable" failure path independent of what's
    # actually on PATH in whatever environment runs this test.
    monkeypatch.setattr("shutil.which", lambda *_args, **_kwargs: None)
    with pytest.raises(DocumentConversionUnavailable, match="no LibreOffice binary"):
        convert_to_pdf(tmp_path / "anything.docx", tmp_path / "out.pdf")
