"""Permission levels an action can require, and the possible decisions."""
from __future__ import annotations

import enum


class PermissionLevel(enum.IntEnum):
    """Higher number = more dangerous / harder to reverse."""

    LOW = 10
    """Read files, search, list directories, create new files, run
    allowlisted code, compute — safe to auto-run."""

    REVIEW = 20
    """Delete files, overwrite existing files, send messages/emails,
    upload, submit, purchase — any irreversible external action. Requires
    explicit approval unless the user has pre-trusted it."""

    RESTRICTED = 30
    """Reserved for actions that should never auto-run under any current
    policy (e.g. modifying Kanna's own permission rules)."""


class Decision(enum.Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"
