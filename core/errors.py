"""Kanna's exception hierarchy.

Every error the agent or a tool raises deliberately (as opposed to an
unexpected bug) should be a `KannaError` subclass so callers can catch
"expected, reportable" failures separately from programming errors.
"""
from __future__ import annotations


class KannaError(Exception):
    """Base class for all deliberate Kanna errors."""


class ConfigError(KannaError):
    """Configuration is missing or invalid."""


class ValidationError(KannaError):
    """Data failed schema validation."""


class PermissionDenied(KannaError):
    """An action was denied by the permission policy."""


class ApprovalRequired(KannaError):
    """An action requires explicit user approval before it can run."""


class SandboxViolation(KannaError):
    """A filesystem path attempted to escape its sandbox root."""


class ToolNotFound(KannaError):
    """No tool is registered under the requested name."""


class ToolExecutionError(KannaError):
    """A tool raised while executing (as opposed to returning a failed result)."""


class CapabilityUnavailable(KannaError):
    """The requested capability has no implementation on this device/environment.

    Raised instead of silently pretending the action succeeded — e.g. no
    computer-control backend is installed, or a required binary is missing.
    """


class LLMUnavailable(KannaError):
    """No usable LLM provider is configured (e.g. missing API key)."""


class VisionUnavailable(KannaError):
    """No usable vision provider is configured (e.g. missing API key or package)."""


class DocumentGenerationUnavailable(KannaError):
    """The package needed to render a document format isn't installed (e.g. python-docx)."""


class BrowserUnavailable(KannaError):
    """No usable browser backend is configured (e.g. playwright or its browser binary missing)."""


class BrowserActionFailed(KannaError):
    """A browser action (navigate/click/fill) could not be completed — e.g. no matching element.

    Distinct from `BrowserUnavailable`: the browser itself works fine,
    but this specific action didn't — the selector didn't match
    anything, the element wasn't interactable, navigation timed out,
    etc. Never silently treated as success.
    """


class PlanningError(KannaError):
    """The planner could not produce a valid plan for the request."""


class AgentBlocked(KannaError):
    """The agent loop cannot proceed without external input (approval, auth, etc.)."""


class FinanceError(KannaError):
    """Base class for finance-subsystem errors."""


class CurrencyMismatch(FinanceError):
    """An operation combined amounts in two different currencies."""


class InvalidMoneyAmount(FinanceError):
    """A monetary amount could not be parsed or was out of range."""
