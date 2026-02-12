"""Regex-based permission engine — pure logic, no I/O, no side effects.

Every tool invocation is an action string: ``tool:<tool_name>:<detail>``.
The engine matches it against a :class:`PermissionProfile` and returns
:data:`ALLOW`, :data:`ASK`, or :data:`DENY`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InvalidPermissionPatternError(ValueError):
    """Raised when a regex pattern in a permission profile is invalid."""


class UnknownPresetError(KeyError):
    """Raised when requesting a preset that does not exist."""


# ---------------------------------------------------------------------------
# Result enum
# ---------------------------------------------------------------------------


class PermissionResult(Enum):
    """Outcome of a permission check."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# ---------------------------------------------------------------------------
# PermissionProfile
# ---------------------------------------------------------------------------


@dataclass
class PermissionProfile:
    """A set of regex patterns that control what an agent may do.

    Patterns are compiled once at construction time.  Invalid patterns
    raise :exc:`InvalidPermissionPatternError` immediately.
    """

    allow: list[str]
    ask: list[str]
    _compiled_allow: list[re.Pattern[str]] = field(init=False, repr=False)
    _compiled_ask: list[re.Pattern[str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            self._compiled_allow = [re.compile(p) for p in self.allow]
        except re.error as exc:
            raise InvalidPermissionPatternError(f"Invalid allow pattern: {exc}") from exc
        try:
            self._compiled_ask = [re.compile(p) for p in self.ask]
        except re.error as exc:
            raise InvalidPermissionPatternError(f"Invalid ask pattern: {exc}") from exc

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {"allow": self.allow, "ask": self.ask}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PermissionProfile:
        """Deserialize from a dict (e.g. from ``agent.json``)."""
        return cls(allow=data["allow"], ask=data["ask"])


# ---------------------------------------------------------------------------
# Engine — stateless check function
# ---------------------------------------------------------------------------


def check(action: str, profile: PermissionProfile) -> PermissionResult:
    """Check *action* against *profile*.

    Matching order: allow → ask → deny.  Uses ``fullmatch`` so the
    entire action string must match the pattern.
    """
    for pattern in profile._compiled_allow:
        if pattern.fullmatch(action):
            return PermissionResult.ALLOW
    for pattern in profile._compiled_ask:
        if pattern.fullmatch(action):
            return PermissionResult.ASK
    return PermissionResult.DENY


# ---------------------------------------------------------------------------
# Action string helper
# ---------------------------------------------------------------------------


def format_action(tool_name: str, detail: str = "") -> str:
    """Build a correctly formatted action string."""
    return f"tool:{tool_name}:{detail}"


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, PermissionProfile] = {
    "open": PermissionProfile(allow=[".*"], ask=[]),
    "standard": PermissionProfile(
        allow=[r"tool:file:.*", r"tool:git:(?!push).*"],
        ask=[r"tool:bash:.*", r"tool:git:push.*"],
    ),
    "locked": PermissionProfile(allow=[r"tool:file:view.*"], ask=[]),
}


def get_preset(name: str) -> PermissionProfile:
    """Return a built-in preset by *name*.

    Raises :exc:`UnknownPresetError` if the name is not recognised.
    """
    try:
        return PRESETS[name]
    except KeyError:
        raise UnknownPresetError(
            f"Unknown permission preset: {name!r}. Available: {', '.join(PRESETS)}"
        ) from None
