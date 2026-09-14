"""The structured result every tool returns.

Tools never return bare strings for anything a caller might need to act
on programmatically — success/failure, files touched, and machine-usable
`data` are always explicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolError:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: ToolError | None = None
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    @classmethod
    def ok(cls, data: dict[str, Any] | None = None, **kwargs: Any) -> "ToolResult":
        return cls(success=True, data=data or {}, **kwargs)

    @classmethod
    def fail(cls, code: str, message: str, details: dict[str, Any] | None = None,
              **kwargs: Any) -> "ToolResult":
        return cls(success=False, error=ToolError(code=code, message=message, details=details or {}), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": None if self.error is None else {
                "code": self.error.code,
                "message": self.error.message,
                "details": self.error.details,
            },
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "files_deleted": self.files_deleted,
            "metadata": self.metadata,
            "duration_ms": self.duration_ms,
        }
