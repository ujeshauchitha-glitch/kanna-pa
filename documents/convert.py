"""DOCX/PPTX -> PDF conversion via LibreOffice's headless mode.

Unlike `docx_writer.py`/`pptx_writer.py`/`pdf_writer.py` (direct Python
libraries rendering from `documents.model.Document`, no external
process), converting an *already-rendered* document into PDF has no
pure-Python equivalent worth trusting — layout fidelity depends on
actually re-laying the document out, which is exactly what a real
office suite does. This shells out to `soffice --headless
--convert-to pdf`, the same real tool a person would run from a
terminal, rather than approximating it.

Lazily checks for the binary (`is_available()`/`shutil.which`) so a
machine without LibreOffice reports `DocumentConversionUnavailable`
honestly instead of crashing or producing a fake/empty PDF.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from core.errors import DocumentConversionUnavailable

_DEFAULT_TIMEOUT_SECONDS = 60


def _soffice_binary() -> str | None:
    return shutil.which("soffice") or shutil.which("libreoffice")


def is_available() -> bool:
    """True if a LibreOffice binary is on PATH. Doesn't guarantee a given
    conversion will succeed (a corrupt input file still fails) — only
    that the capability itself exists on this machine."""
    return _soffice_binary() is not None


def convert_to_pdf(source: Path, dest: Path, *, timeout: int = _DEFAULT_TIMEOUT_SECONDS) -> None:
    """Convert the DOCX/PPTX (or anything LibreOffice can read) at `source`
    into a PDF at exactly `dest`.

    Raises `DocumentConversionUnavailable` if no LibreOffice binary is
    found, `source` doesn't exist, the conversion times out, or soffice
    exits without producing a PDF — never leaves a partial or corrupt
    file at `dest` in any failure case.
    """
    binary = _soffice_binary()
    if binary is None:
        raise DocumentConversionUnavailable(
            "no LibreOffice binary ('soffice'/'libreoffice') found on PATH; "
            "install LibreOffice to enable DOCX/PPTX -> PDF conversion"
        )
    if not source.exists():
        raise DocumentConversionUnavailable(f"source file does not exist: {source}")

    # soffice always names its output <source-stem>.pdf in --outdir, never
    # `dest` directly, so it's converted into a scratch dir first and moved
    # into place. -env:UserInstallation points it at a fresh, isolated
    # profile dir per call — without this, concurrent/rapid invocations
    # collide on soffice's shared user profile lock ("Fatal Error: could
    # not obtain lock"), a well-known headless-soffice gotcha that would
    # otherwise make this flaky under real use (or in a test suite running
    # more than one conversion).
    with tempfile.TemporaryDirectory(prefix="kanna-soffice-out-") as out_dir, \
         tempfile.TemporaryDirectory(prefix="kanna-soffice-profile-") as profile_dir:
        try:
            completed = subprocess.run(
                [binary, "--headless", "--norestore",
                 f"-env:UserInstallation=file://{profile_dir}",
                 "--convert-to", "pdf", "--outdir", out_dir, str(source)],
                capture_output=True, text=True, timeout=timeout, shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise DocumentConversionUnavailable(
                f"soffice conversion timed out after {timeout}s"
            ) from exc
        except OSError as exc:
            raise DocumentConversionUnavailable(f"could not run soffice: {exc}") from exc

        produced = Path(out_dir) / f"{source.stem}.pdf"
        if completed.returncode != 0 or not produced.exists():
            detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
            raise DocumentConversionUnavailable(f"soffice conversion failed: {detail}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), str(dest))
