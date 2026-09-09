"""`kanna document generate ...` — build a DOCX/PPTX/PDF from a JSON content file.

The content shape (title/subtitle/author/sections/path) is exactly the
`document_generate_*` tools' input schema (`documents/tools.py`) — the
same JSON file works whether it's handed to this CLI command or to an
agent-loop plan step.
"""
from __future__ import annotations

import argparse
import json
import sys

from core.bootstrap import Kanna
from core.errors import DocumentGenerationUnavailable, SandboxViolation
from documents.docx_writer import render_docx
from documents.model import build_document
from documents.pdf_writer import render_pdf
from documents.pptx_writer import render_pptx

_RENDERERS = {"docx": render_docx, "pptx": render_pptx, "pdf": render_pdf}


def register(subparsers: argparse._SubParsersAction) -> None:
    doc_parser = subparsers.add_parser("document", help="Generate DOCX/PPTX/PDF documents")
    doc_sub = doc_parser.add_subparsers(dest="document_command", required=True)

    gen_p = doc_sub.add_parser(
        "generate", help="Generate a document from a JSON content file "
                          "({title, subtitle?, author?, path, sections: [...]})")
    gen_p.add_argument("json_file", help="Path to the JSON content file (read from the local "
                                          "filesystem, not the sandbox)")
    gen_p.add_argument("--format", required=True, choices=sorted(_RENDERERS))
    gen_p.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting an existing file at the content's 'path'")
    gen_p.set_defaults(func=_cmd_generate)


def _cmd_generate(args: argparse.Namespace, kanna: Kanna) -> int:
    try:
        with open(args.json_file, encoding="utf-8") as fh:
            content = json.load(fh)
    except OSError as exc:
        print(f"error: could not read {args.json_file}: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"error: {args.json_file} is not valid JSON: {exc}", file=sys.stderr)
        return 1

    if "path" not in content or "title" not in content:
        print("error: content JSON must include at least 'title' and 'path'", file=sys.stderr)
        return 1

    try:
        resolved = kanna.sandbox.resolve(content["path"])
    except SandboxViolation as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if resolved.exists() and not args.overwrite:
        print(f"error: '{resolved}' already exists; pass --overwrite to replace it", file=sys.stderr)
        return 1

    document = build_document(content)
    try:
        _RENDERERS[args.format](document, resolved)
    except DocumentGenerationUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Generated {resolved} ({resolved.stat().st_size} bytes).")
    return 0
