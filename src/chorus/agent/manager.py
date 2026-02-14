"""Agent lifecycle manager — orchestrates directory, database, and configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from chorus.agent.directory import AgentDirectory, read_agent_json, write_agent_json
from chorus.models import Agent, AgentNotFoundError, validate_agent_name

if TYPE_CHECKING:
    from chorus.config import BotConfig, GlobalConfig
    from chorus.storage.db import Database

logger = logging.getLogger("chorus.agent.manager")

CONFIGURABLE_KEYS = {"system_prompt", "model", "permissions", "web_search"}


class AgentManager:
    """Coordinates agent CRUD across filesystem and database."""

    def __init__(
        self,
        directory: AgentDirectory,
        db: Database,
        global_config: GlobalConfig | None = None,
        bot_config: BotConfig | None = None,
    ) -> None:
        self._directory = directory
        self._db = db
        self._global_config = global_config
        self._bot_config = bot_config

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

        # Refine system prompt via cheap LLM
        if self._bot_config is not None:
            from chorus.agent.prompt_refinement import refine_system_prompt

            user_description = (overrides or {}).get("system_prompt")
            template_prompt = agent_data.get("system_prompt", "")
            try:
                agent_data["system_prompt"] = await refine_system_prompt(
                    agent_name=name,
                    user_description=user_description,
                    template_prompt=template_prompt,
                    config=self._bot_config,
                )
            except Exception:
                logger.exception("Prompt refinement failed for agent %s", name)
                # If user provided an explicit override, use that; otherwise keep template
                if user_description:
                    agent_data["system_prompt"] = user_description

        # Apply global defaults for fields not explicitly overridden
        if self._global_config is not None:
            override_keys = set(overrides or {})
            if "model" not in override_keys and not agent_data.get("model"):
                agent_data["model"] = self._global_config.default_model
            if "permissions" not in override_keys:
                agent_data["permissions"] = self._global_config.default_permissions

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

    async def get_agent_by_channel(self, channel_id: int) -> Agent | None:
        """Look up an agent by its Discord channel ID."""
        row = await self._db.get_agent_by_channel(channel_id)
        if row is None:
            return None
        return Agent.from_dict(row)

    async def configure(self, name: str, key: str, value: str) -> None:
        """Update a single configuration key in an agent's agent.json."""
        if key not in CONFIGURABLE_KEYS:
            raise ValueError(f"Cannot configure key: {key!r}. Allowed: {CONFIGURABLE_KEYS}")

        agent_path = self._directory.get(name)
        if agent_path is None:
            raise AgentNotFoundError(f"Agent {name!r} not found")

        config_path = agent_path / "agent.json"
        data = read_agent_json(config_path)
        if key == "web_search":
            data[key] = value.lower() in ("true", "1", "yes")
        else:
            data[key] = value
        write_agent_json(config_path, data)
        logger.info("Configured agent %s: %s = %s", name, key, value)
