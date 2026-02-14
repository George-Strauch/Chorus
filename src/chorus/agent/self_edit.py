"""Agent self-modification — edit system prompt, docs, permissions, model.

All edits are logged to the audit table and take effect on the next LLM call.
Permission editing is gated by the invoking user's role (is_admin flag).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from chorus.config import validate_model_available
from chorus.permissions.engine import PRESETS
from chorus.tools.file_ops import PathTraversalError, resolve_in_workspace

if TYPE_CHECKING:
    from pathlib import Path

    from chorus.storage.db import Database

logger = logging.getLogger("chorus.agent.self_edit")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SelfEditResult:
    """Structured result from a self-edit operation."""

    success: bool
    edit_type: str
    old_value: str
    new_value: str
    message: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "success": self.success,
            "edit_type": self.edit_type,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "message": self.message,
        }
        if self.error is not None:
            d["error"] = self.error
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_AUDIT_TRUNCATE = 500


def _truncate(value: str, limit: int = _AUDIT_TRUNCATE) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON atomically via tmp-file + rename."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=4) + "\n")
    tmp.rename(path)


def _read_agent_json(workspace: Path) -> tuple[Path, dict[str, Any]]:
    """Read agent.json from workspace.parent and return (path, data)."""
    agent_json = workspace.parent / "agent.json"
    data = json.loads(agent_json.read_text())
    return agent_json, data


# ---------------------------------------------------------------------------
# Core self-edit functions
# ---------------------------------------------------------------------------


async def edit_system_prompt(
    new_prompt: str,
    *,
    workspace: Path,
    agent_name: str,
    db: Database | None = None,
) -> SelfEditResult:
    """Update the agent's system prompt in agent.json."""
    if not new_prompt.strip():
        return SelfEditResult(
            success=False,
            edit_type="system_prompt",
            old_value="",
            new_value="",
            message="System prompt cannot be empty.",
            error="empty_prompt",
        )

    agent_json, data = _read_agent_json(workspace)
    old_prompt = data.get("system_prompt", "")
    data["system_prompt"] = new_prompt
    _atomic_write_json(agent_json, data)

    if db is not None:
        await db.log_self_edit(
            agent_name=agent_name,
            edit_type="system_prompt",
            old_value=old_prompt,
            new_value=new_prompt,
        )

    return SelfEditResult(
        success=True,
        edit_type="system_prompt",
        old_value=old_prompt,
        new_value=new_prompt,
        message="System prompt updated.",
    )


async def edit_docs(
    path: str,
    content: str,
    *,
    workspace: Path,
    agent_name: str,
    db: Database | None = None,
) -> SelfEditResult:
    """Create or update a file in the agent's docs/ directory."""
    docs_dir = workspace.parent / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    try:
        resolved = resolve_in_workspace(docs_dir, path)
    except PathTraversalError:
        return SelfEditResult(
            success=False,
            edit_type="docs",
            old_value="",
            new_value="",
            message=f"Path traversal denied: {path!r}",
            error="path_traversal",
        )

    old_value = ""
    if resolved.exists():
        old_value = resolved.read_text(encoding="utf-8")

    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")

    if db is not None:
        await db.log_self_edit(
            agent_name=agent_name,
            edit_type="docs",
            old_value=old_value,
            new_value=content,
        )

    action = "updated" if old_value else "created"
    return SelfEditResult(
        success=True,
        edit_type="docs",
        old_value=old_value,
        new_value=content,
        message=f"Doc file {action}: {path}",
    )


async def edit_permissions(
    profile: str,
    *,
    workspace: Path,
    agent_name: str,
    is_admin: bool = False,
    db: Database | None = None,
) -> SelfEditResult:
    """Update the agent's permission profile in agent.json.

    ``locked`` and ``standard`` are allowed for anyone.
    ``open`` requires ``is_admin=True``.
    """
    # Validate preset name
    if profile not in PRESETS:
        return SelfEditResult(
            success=False,
            edit_type="permissions",
            old_value="",
            new_value=profile,
            message=f"Unknown permission preset: {profile!r}. "
            f"Available: {', '.join(PRESETS)}",
            error="unknown_preset",
        )

    # Role gating
    if profile == "open" and not is_admin:
        return SelfEditResult(
            success=False,
            edit_type="permissions",
            old_value="",
            new_value=profile,
            message="Only admins (Manage Server) can set 'open' permissions.",
            error="insufficient_role",
        )

    agent_json, data = _read_agent_json(workspace)
    old_profile = data.get("permissions", "standard")
    data["permissions"] = profile
    _atomic_write_json(agent_json, data)

    if db is not None:
        await db.update_agent_field(agent_name, "permissions", profile)
        await db.log_self_edit(
            agent_name=agent_name,
            edit_type="permissions",
            old_value=old_profile,
            new_value=profile,
        )

    return SelfEditResult(
        success=True,
        edit_type="permissions",
        old_value=old_profile,
        new_value=profile,
        message=f"Permissions updated to '{profile}'.",
    )


def _resolve_short_model_name(model: str, chorus_home: Path | None) -> str:
    """Resolve a short model name (e.g. 'opus', 'haiku') to a full model ID.

    Returns the original name if no match is found or no cache is available.
    """
    if chorus_home is None:
        return model
    from chorus.llm.discovery import get_cached_models

    cached = get_cached_models(chorus_home)
    if not cached:
        return model

    # If exact match exists, return as-is
    if model in cached:
        return model

    # Try substring match (case-insensitive)
    needle = model.lower()
    for m in cached:
        if needle in m.lower():
            return m

    return model


async def edit_model(
    model: str,
    *,
    workspace: Path,
    agent_name: str,
    chorus_home: Path | None = None,
    db: Database | None = None,
) -> SelfEditResult:
    """Update the agent's model in agent.json.

    Supports short names (e.g. 'opus' → 'claude-opus-4-20250514') via fuzzy matching.
    Validates against the available models cache if *chorus_home* is provided.
    """
    # Resolve short names first
    resolved = _resolve_short_model_name(model, chorus_home)

    if chorus_home is not None and not validate_model_available(resolved, chorus_home):
        return SelfEditResult(
            success=False,
            edit_type="model",
            old_value="",
            new_value=model,
            message=f"Model {model!r} is not available. "
            "Run /settings validate-keys to refresh the model list.",
            error="unavailable_model",
        )

    # Use the resolved name for the actual update
    model = resolved

    agent_json, data = _read_agent_json(workspace)
    old_model = data.get("model") or ""
    data["model"] = model
    _atomic_write_json(agent_json, data)

    if db is not None:
        await db.update_agent_field(agent_name, "model", model)
        await db.log_self_edit(
            agent_name=agent_name,
            edit_type="model",
            old_value=old_model,
            new_value=model,
        )

    return SelfEditResult(
        success=True,
        edit_type="model",
        old_value=old_model,
        new_value=model,
        message=f"Model updated to '{model}'.",
    )


async def edit_web_search(
    enabled: bool,
    *,
    workspace: Path,
    agent_name: str,
    db: Database | None = None,
) -> SelfEditResult:
    """Toggle the agent's web search capability in agent.json."""
    agent_json, data = _read_agent_json(workspace)
    old_value = str(data.get("web_search", False)).lower()
    data["web_search"] = enabled
    _atomic_write_json(agent_json, data)

    new_value = str(enabled).lower()

    if db is not None:
        await db.log_self_edit(
            agent_name=agent_name,
            edit_type="web_search",
            old_value=old_value,
            new_value=new_value,
        )

    state = "enabled" if enabled else "disabled"
    return SelfEditResult(
        success=True,
        edit_type="web_search",
        old_value=old_value,
        new_value=new_value,
        message=f"Web search {state}.",
    )
