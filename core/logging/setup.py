"""Logging setup with automatic secret redaction.

Kanna's logs may end up attached to bug reports, so anything that looks
like an API key or token is scrubbed before a record is emitted, not just
"by convention" at each call site.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{10,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)(['\"]?\s*[:=]\s*['\"]?)([A-Za-z0-9\-_./+]{8,})"),
]

_REDACTED = "[REDACTED]"


def redact(text: str) -> str:
    out = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            out = pattern.sub(lambda m: m.group(1) + m.group(2) + _REDACTED, out)
        else:
            out = pattern.sub(_REDACTED, out)
    return out


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(str(record.getMessage()))
        record.args = ()
        return True


_CONFIGURED = False


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> logging.Logger:
    """Configure the root `kanna` logger once. Safe to call repeatedly."""
    global _CONFIGURED
    logger = logging.getLogger("kanna")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if _CONFIGURED:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"
    )

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(fmt)
    stream_handler.addFilter(RedactionFilter())
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "kanna.log")
        file_handler.setFormatter(fmt)
        file_handler.addFilter(RedactionFilter())
        logger.addHandler(file_handler)

    logger.propagate = False
    _CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"kanna.{name}")
