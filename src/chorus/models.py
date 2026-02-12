"""Data models for Chorus agents, sessions, and related types."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

AGENT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}[a-z0-9]$")

DEFAULT_SYSTEM_PROMPT = (
    "You are a general-purpose AI agent. You have access to a workspace directory "
    "where you can create, edit, and view files, run commands, and manage a git "
    "repository. Use your tools to accomplish tasks. Maintain notes about your "
    "workspace in your docs/ directory."
)


class InvalidAgentNameError(ValueError):
    """Raised when an agent name doesn't match the required pattern."""


class AgentExistsError(Exception):
    """Raised when trying to create an agent that already exists."""


class AgentNotFoundError(Exception):
    """Raised when an agent is not found."""


def validate_agent_name(name: str) -> None:
    """Validate an agent name against the naming rules.

    Names must be 2-32 chars, lowercase alphanumeric + hyphens,
    starting and ending with alphanumeric.
    """
    if not AGENT_NAME_PATTERN.match(name):
        raise InvalidAgentNameError(
            f"Invalid agent name: {name!r}. Names must be 2-32 lowercase "
            f"alphanumeric characters or hyphens, starting and ending with alphanumeric."
        )


@dataclass
class Agent:
    """Represents a Chorus agent and its configuration."""

    name: str
    channel_id: int
    model: str | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    permissions: str = "standard"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        """Construct an Agent from a dict (e.g. agent.json or DB row)."""
        return cls(
            name=data["name"],
            channel_id=data.get("channel_id", 0),
            model=data.get("model"),
            system_prompt=data.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            permissions=data.get("permissions", "standard"),
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the agent to a dict suitable for JSON."""
        return {
            "name": self.name,
            "channel_id": self.channel_id,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "permissions": self.permissions,
            "created_at": self.created_at,
        }
