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
without `overwrite=true`. Registered at REVIEW, downgraded to auto-allow for the create case by the
same policy rule `fs_write_file` uses — see `core/bootstrap.py::default_policy` (generalized in this
pass to cover all four create-or-overwrite tools by name, instead of a `fs_write_file`-only rule).

## CLI

```bash
kanna document generate content.json --format docx   # or pptx, pdf
kanna document generate content.json --format pdf --overwrite
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
`test_documents_tools.py` (sandboxing, overwrite protection, all three formats), and
`test_bootstrap.py` (the generalized create-or-overwrite policy rule covers all four tools, not just
`fs_write_file`). All fully offline — no network, no LLM, no `ANTHROPIC_API_KEY` — since
`python-docx`/`python-pptx`/`reportlab` are deterministic local libraries; tests use
`pytest.importorskip` so the suite degrades gracefully (skips, doesn't fail) if `kanna[documents]`
isn't installed, though `pip install -e ".[dev]"` includes it.

Manually verified end-to-end through the real `bootstrap()`/registry path and the CLI, including the
overwrite-protection round trip (`generate` → refuse → `--overwrite` succeeds).

## Known limitations

- No PPTX design/theming beyond python-pptx's default template — no custom fonts, colors, or layouts
  beyond what `Presentation()`'s built-in layouts provide.
- PDF layout is simple platypus flowables (headings, paragraphs, bullets-as-text, grid tables) — no
  page headers/footers, multi-column layout, or embedded images yet.
- No DOCX/PPTX → PDF conversion tool, even though LibreOffice (`soffice`) is present in this
  environment — the direct `reportlab` PDF path was chosen instead so PDF generation doesn't depend on
  what else happens to be installed on a given host. A `soffice`-based converter (reusing
  `tools/process/run_process.py`'s controlled-execution pattern) remains a reasonable future addition
  for cases needing exact DOCX-fidelity PDF output.
- No reading/editing of existing DOCX/PPTX/PDF files — generation only.
