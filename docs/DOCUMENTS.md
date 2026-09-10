# Documents

## Status: real, offline, no LLM required at render time

Phase 2 adds `documents/`: DOCX, PPTX, and PDF generation from one shared structured content model.
Unlike `vision/` and the LLM planner, none of this needs `ANTHROPIC_API_KEY` or the network — every
renderer is a pure, deterministic Python library call (`python-docx`, `python-pptx`, `reportlab`).
An LLM is only involved, if at all, *before* this module — turning "write me a report about X" into
the structured content `documents` renders — never inside rendering itself.

## The shared content model (`documents/model.py`)

```python
Document(title, subtitle=None, author=None, sections: list[Section])
Section(heading=None, level=1, paragraphs=[], bullets=[], table: TableData | None)
TableData(headers: list[str], rows: list[list[str]])
```

One `Document` renders to any of the three formats — callers describe content once. This mirrors how
`finance.money.Money`/`finance.models.Transaction` feed every finance entry path from one model. Raw
tool/CLI input (JSON-shaped dicts) becomes a `Document` via `build_document()`, kept separate from
validation (`documents/tools.py`'s `DOCUMENT_INPUT_SCHEMA`) the same way `finance/nlp.py`'s parsing is
kept separate from its callers.

## Writers

| Module | Library | Notes |
|---|---|---|
| `documents/docx_writer.py` | `python-docx` | Title/Subtitle styles, `Heading 1..9`, `List Bullet`, `Table Grid` |
| `documents/pptx_writer.py` | `python-pptx` | One slide per top-level section (slides are flat — `Section.level` doesn't apply); bullets preferred over paragraphs for slide body text; a table-only section uses the "Title Only" layout so it still gets its heading (see below) |
| `documents/pdf_writer.py` | `reportlab` (platypus flowables) | `Heading1..6` styles by clamped `Section.level`; all text is XML-escaped before being handed to `Paragraph` (reportlab treats its content as minimal markup — unescaped `&`/`<`/`>` would raise) |

Each writer raises `core.errors.DocumentGenerationUnavailable` — not a crash, not a silent empty
file — if its library isn't installed (`pip install kanna[documents]`).

### A bug this module's tests specifically guard against

The PPTX writer originally routed a section with *only* a table (no bullets/paragraphs) onto
python-pptx's "Blank" layout, which has no title placeholder — the section's heading was silently
dropped. Fixed by routing that case to the "Title Only" layout instead. `test_render_pptx_...` in
`tests/test_documents_pptx.py` asserts the heading survives specifically for a table-only section, so
this can't silently regress.

## Tools (`documents/tools.py`)

`document_generate_docx`, `document_generate_pptx`, `document_generate_pdf` — one per format, sharing
`DOCUMENT_INPUT_SCHEMA` (`title`, optional `subtitle`/`author`, `path`, `sections`, `overwrite`) and a
common `execute()` that resolves `path` through the sandbox and refuses to overwrite an existing file
without `overwrite=true`. `document_convert_to_pdf` (below) follows the identical sandboxing/overwrite
pattern with its own `source_path`/`dest_path` schema. All four are registered at REVIEW, downgraded
to auto-allow for the create case by the same policy rule `fs_write_file` uses — see
`core/bootstrap.py::default_policy` (covers all five create-or-overwrite tools by name, instead of a
`fs_write_file`-only rule).

## DOCX/PPTX → PDF conversion (`documents/convert.py`)

`convert_to_pdf(source, dest)` shells out to `soffice --headless --convert-to pdf` — LibreOffice's own
converter, the same one a person would use from a terminal — rather than approximating DOCX/PPTX
layout fidelity in Python. This is a different tradeoff from the direct-`reportlab` PDF writer above:
`document_generate_pdf` renders *new* content and has zero external dependencies; `convert_to_pdf`
turns an *already-rendered* DOCX/PPTX into a faithful PDF of that exact document, which only a real
layout engine can do credibly, at the cost of depending on LibreOffice being installed. Both exist
because they solve different problems, not because one replaces the other.

Two real issues came up validating this against actual LibreOffice, not just calling it and assuming
it works:

1. **A minimal LibreOffice install can't convert anything.** This build environment initially had only
   `libreoffice-core`/`libreoffice-common` installed — `soffice` itself runs, but every conversion
   failed with `Error: source file could not be loaded`, for *any* input, including a plain `.txt`
   file. `strace` traced it to a missing `libswdlo.so` (the Writer import/export filter, part of the
   separate `libreoffice-writer` package) — `libreoffice-impress` is the equivalent for PPTX. Neither
   ships with `libreoffice-core` alone. Fixed by installing both — `.github/workflows/tests.yml` now
   does the same, so CI exercises real conversions rather than perpetually failing or skipping.
2. **Headless soffice needs an isolated profile per call.** Without `-env:UserInstallation=file://...`
   pointing at a fresh temp directory, concurrent or rapid successive `soffice` invocations collide on
   its shared user-profile lock (`Fatal Error: could not obtain lock`) — a well-known headless-soffice
   gotcha that would otherwise make this flaky under real use or in a test suite running more than one
   conversion. `convert_to_pdf()` creates a throwaway profile dir (and output dir — soffice always
   names its own output `<source-stem>.pdf`, never the caller's `dest`, so the real file gets moved
   into place afterward) per call, cleaned up automatically.

`is_available()` (`shutil.which("soffice")` or `"libreoffice"`) lets a caller check without attempting
a conversion; `convert_to_pdf()` itself raises `core.errors.DocumentConversionUnavailable` — never a
crash or a silently-empty PDF — if the binary is missing, `source` doesn't exist, the conversion times
out, or `soffice` exits without producing a PDF.

**Worth knowing:** LibreOffice's format detection is content-based, not extension-based, and it is
*very* tolerant — even a file with a misleading extension, or one containing effectively random bytes,
usually "succeeds" by falling back to some interpretation of the content rather than failing outright.
Don't rely on "feed it garbage" as a way to test the failure path; the reliable ways to exercise
`DocumentConversionUnavailable` are a genuinely missing source file, a missing binary, or a timeout —
see `tests/test_documents_convert.py`.

## CLI

```bash
kanna document generate content.json --format docx   # or pptx, pdf
kanna document generate content.json --format pdf --overwrite
kanna document convert report.docx report.pdf
kanna document convert report.docx report.pdf --overwrite
```

`content.json` is exactly the tools' input shape:

```json
{
  "title": "Weekly Status Report",
  "subtitle": "Week 12",
  "author": "Kanna",
  "path": "status.pdf",
  "sections": [
    {"heading": "Highlights", "bullets": ["Shipped vision", "Started document generation"]},
    {"heading": "Metrics", "table": {"headers": ["Metric", "Value"], "rows": [["Tests", "199"]]}}
  ]
}
```

`path` is resolved through the same sandbox every filesystem tool uses.

## Why there's no rule-based-planner pattern for this

`core/planner/rule_based.py` recognizes single-sentence intents it can regex into a complete tool
call. "Write me a status report" can't become a `document_generate_docx` call that way — the actual
section content has to come from somewhere, and synthesizing it is exactly an LLM's job. So document
generation is reachable via the tool registry directly (an `LLMPlanner`-produced plan can call these
tools — their schema-validated JSON args are exactly what `LLMPlanner` already knows how to produce
and validate) and via the CLI's explicit JSON-file input; there's no natural-language shortcut for it
in Phase 2.

## Testing

`tests/test_documents_model.py` (pure `build_document()` conversion), `test_documents_docx.py`,
`test_documents_pptx.py` (including the table-only-heading regression above), `test_documents_pdf.py`
(including a regression test for the XML-escaping — an unescaped `&`/`<`/`>` used to crash rendering),
`test_documents_tools.py` (sandboxing, overwrite protection, all three formats, plus
`document_convert_to_pdf`'s tool layer via a monkeypatched `convert_to_pdf` so that coverage doesn't
need LibreOffice installed), `test_documents_convert.py` (the real `convert_to_pdf()` — real DOCX/PPTX
input converted by real `soffice`, the move-into-place logic actually landing the file at the exact
`dest` path rather than wherever `soffice` names its own output, missing source, timeout, and a
`shutil.which`-monkeypatched "no binary" case that runs regardless of whether LibreOffice is actually
installed), and `test_bootstrap.py` (the generalized create-or-overwrite policy rule covers all five
tools, not just `fs_write_file`). Generation is fully offline — no network, no LLM, no
`ANTHROPIC_API_KEY` — since `python-docx`/`python-pptx`/`reportlab` are deterministic local libraries;
those tests use `pytest.importorskip` so the suite degrades gracefully (skips, doesn't fail) if
`kanna[documents]` isn't installed. `test_documents_convert.py`'s real-conversion tests similarly
`skipif` when no LibreOffice binary is on PATH. Neither skips in this sandbox or in CI, where
`libreoffice-writer`/`libreoffice-impress` are installed alongside `kanna[dev]`.

Manually verified end-to-end through the real `bootstrap()`/registry path and the CLI, including the
overwrite-protection round trip (`generate` → refuse → `--overwrite` succeeds) for both `generate` and
`convert`.

## Known limitations

- No PPTX design/theming beyond python-pptx's default template — no custom fonts, colors, or layouts
  beyond what `Presentation()`'s built-in layouts provide.
- PDF layout is simple platypus flowables (headings, paragraphs, bullets-as-text, grid tables) — no
  page headers/footers, multi-column layout, or embedded images yet.
- DOCX/PPTX → PDF conversion depends on LibreOffice being installed on the host — `is_available()`
  reports that honestly rather than the tool crashing or silently doing nothing, but there's no
  fallback conversion path if it's absent (the direct-`reportlab` `document_generate_pdf` path remains
  available regardless, for *new* content rather than converting existing files).
- No reading/editing of existing DOCX/PPTX/PDF files — generation and DOCX/PPTX→PDF conversion only,
  no other conversions (e.g. PDF→DOCX).
