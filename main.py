#!/usr/bin/env python3
"""Kanna entry point: `python main.py <command> ...` (also installed as the `kanna` console script)."""
from __future__ import annotations

import sys

from interfaces.cli.app import main

if __name__ == "__main__":
    sys.exit(main())
