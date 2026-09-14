"""Tiny platform-specific "reveal in file manager" helper, shared by the
live output panel (app.py) and the task history dialog — one definition
instead of two copies drifting apart.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def show_in_folder(path: str) -> None:
    """Open the OS file manager at `path`'s parent directory. Raises
    `OSError` on failure — callers show that to the user rather than
    swallowing it, since "nothing happened" with no explanation is worse
    than a shown error."""
    parent = Path(path).parent
    if sys.platform == "win32":
        os.startfile(str(parent))
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(parent)])
