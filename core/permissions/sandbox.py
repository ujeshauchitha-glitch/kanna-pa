"""Filesystem sandboxing.

Every filesystem tool must resolve paths through a `Sandbox` before
touching disk. A path that resolves (after following `..` and symlinks)
outside every configured root is rejected — this is the one thing that
protects the rest of the system from a prompt-injected "read
../../../etc/shadow".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from core.errors import SandboxViolation

logger = logging.getLogger("kanna.permissions.sandbox")


@dataclass
class Sandbox:
    roots: tuple[Path, ...]

    def __init__(self, roots: list[str] | list[Path]) -> None:
        resolved = tuple(Path(r).expanduser().resolve() for r in roots)
        if not resolved:
            raise SandboxViolation("Sandbox configured with no roots")
        object.__setattr__(self, "roots", resolved)

    def resolve(self, path: str | Path) -> Path:
        """Resolve `path` and verify it stays within a sandbox root.

        Accepts absolute paths or paths relative to the first root.
        Symlinks are resolved before the containment check, so a symlink
        inside the sandbox pointing outside it is still caught.
        """
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.roots[0] / candidate

        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError) as exc:  # pragma: no cover - defensive
            raise SandboxViolation(f"Could not resolve path: {path}") from exc

        for root in self.roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue

        logger.warning("sandbox violation: %s resolved to %s, outside roots %s", path, resolved, self.roots)
        raise SandboxViolation(f"Path '{path}' resolves outside the sandbox: {resolved}")

    def is_within(self, path: str | Path) -> bool:
        try:
            self.resolve(path)
            return True
        except SandboxViolation:
            return False
