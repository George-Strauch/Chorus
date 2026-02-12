"""Agent lifecycle manager — orchestrates directory, database, and configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from chorus.agent.directory import AgentDirectory, read_agent_json, write_agent_json
from chorus.models import Agent, AgentNotFoundError, validate_agent_name

if TYPE_CHECKING:
    from chorus.storage.db import Database

logger = logging.getLogger("chorus.agent.manager")

CONFIGURABLE_KEYS = {"system_prompt", "model", "permissions"}


class AgentManager:
    """Coordinates agent CRUD across filesystem and database."""

    def __init__(self, directory: AgentDirectory, db: Database) -> None:
        self._directory = directory
        self._db = db

    async def create(
        self,
        name: str,
        guild_id: int,
        channel_id: int,
        overrides: dict[str, Any] | None = None,
    ) -> Agent:
        """Create a new agent: validate name, copy template, register in DB."""
        validate_agent_name(name)
        agent_path = self._directory.create(name, overrides)
        agent_data = read_agent_json(agent_path / "agent.json")
        agent_data["channel_id"] = channel_id

        # Write channel_id back to agent.json
        write_agent_json(agent_path / "agent.json", agent_data)

        await self._db.register_agent(
            name=name,
            channel_id=channel_id,
            guild_id=guild_id,
            model=agent_data.get("model"),
            permissions=agent_data.get("permissions", "standard"),
            created_at=agent_data["created_at"],
        )

        logger.info("Created agent %s (channel=%d, guild=%d)", name, channel_id, guild_id)
        return Agent.from_dict(agent_data)

    async def destroy(self, name: str, keep_files: bool = False) -> None:
        """Destroy an agent: remove directory and DB row."""
        if self._directory.get(name) is None:
            raise AgentNotFoundError(f"Agent {name!r} not found")
        self._directory.destroy(name, keep_files=keep_files)
        await self._db.remove_agent(name)
        logger.info("Destroyed agent %s (keep_files=%s)", name, keep_files)

    async def list_agents(self, guild_id: int | None = None) -> list[Agent]:
        """List all agents, optionally filtered by guild."""
        rows = await self._db.list_agents(guild_id)
        return [Agent.from_dict(row) for row in rows]

    async def configure(self, name: str, key: str, value: str) -> None:
        """Update a single configuration key in an agent's agent.json."""
        if key not in CONFIGURABLE_KEYS:
            raise ValueError(f"Cannot configure key: {key!r}. Allowed: {CONFIGURABLE_KEYS}")

        agent_path = self._directory.get(name)
        if agent_path is None:
            raise AgentNotFoundError(f"Agent {name!r} not found")

        config_path = agent_path / "agent.json"
        data = read_agent_json(config_path)
        data[key] = value
        write_agent_json(config_path, data)
        logger.info("Configured agent %s: %s = %s", name, key, value)
