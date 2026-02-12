"""Permission engine — regex-based allow/ask/deny for agent tool actions."""

from chorus.permissions.engine import (
    PRESETS,
    InvalidPermissionPatternError,
    PermissionProfile,
    PermissionResult,
    UnknownPresetError,
    check,
    format_action,
    get_preset,
)

__all__ = [
    "PRESETS",
    "InvalidPermissionPatternError",
    "PermissionProfile",
    "PermissionResult",
    "UnknownPresetError",
    "check",
    "format_action",
    "get_preset",
]
