"""Tests for chorus.agent.manager — agent lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from chorus.agent.directory import AgentDirectory
from chorus.agent.manager import AgentManager
from chorus.models import AgentExistsError, AgentNotFoundError, InvalidAgentNameError
from chorus.storage.db import Database

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    db_path = tmp_path / "db" / "chorus.db"
    database = Database(db_path)
    await database.init()
    return database


@pytest.fixture
def agent_directory(tmp_chorus_home: Path, tmp_template: Path) -> AgentDirectory:
    return AgentDirectory(tmp_chorus_home, tmp_template)


@pytest.fixture
def manager(agent_directory: AgentDirectory, db: Database) -> AgentManager:
    return AgentManager(agent_directory, db)


GUILD_ID = 123456789


class TestAgentCreate:
    async def test_create_agent_full_lifecycle(self, manager: AgentManager) -> None:
        agent = await manager.create("test-bot", guild_id=GUILD_ID, channel_id=100)
        assert agent.name == "test-bot"
        assert agent.channel_id == 100

        # Verify it's in DB
        agents = await manager.list_agents(guild_id=GUILD_ID)
        assert len(agents) == 1
        assert agents[0].name == "test-bot"

    async def test_create_agent_with_overrides(self, manager: AgentManager) -> None:
        overrides = {
            "system_prompt": "Custom prompt",
            "model": "gpt-4o",
            "permissions": "open",
        }
        agent = await manager.create(
            "test-bot", guild_id=GUILD_ID, channel_id=100, overrides=overrides
        )
        assert agent.system_prompt == "Custom prompt"
        assert agent.model == "gpt-4o"
        assert agent.permissions == "open"

    async def test_create_agent_rejects_duplicate_name(self, manager: AgentManager) -> None:
        await manager.create("test-bot", guild_id=GUILD_ID, channel_id=100)
        with pytest.raises(AgentExistsError):
            await manager.create("test-bot", guild_id=GUILD_ID, channel_id=200)

    async def test_create_agent_rejects_invalid_name(self, manager: AgentManager) -> None:
        with pytest.raises(InvalidAgentNameError):
            await manager.create("BAD NAME!", guild_id=GUILD_ID, channel_id=100)


class TestAgentDestroy:
    async def test_destroy_agent_cleans_up(self, manager: AgentManager) -> None:
        await manager.create("test-bot", guild_id=GUILD_ID, channel_id=100)
        await manager.destroy("test-bot")
        agents = await manager.list_agents(guild_id=GUILD_ID)
        assert len(agents) == 0

    async def test_destroy_agent_keep_files(
        self, manager: AgentManager, tmp_chorus_home: Path
    ) -> None:
        await manager.create("test-bot", guild_id=GUILD_ID, channel_id=100)
        await manager.destroy("test-bot", keep_files=True)
        # DB row removed
        agents = await manager.list_agents(guild_id=GUILD_ID)
        assert len(agents) == 0
        # Files moved to trash
        assert (tmp_chorus_home / ".trash" / "test-bot").exists()

    async def test_destroy_nonexistent_raises(self, manager: AgentManager) -> None:
        with pytest.raises(AgentNotFoundError):
            await manager.destroy("nonexistent")


class TestAgentList:
    async def test_list_agents_returns_all(self, manager: AgentManager) -> None:
        await manager.create("alpha-bot", guild_id=GUILD_ID, channel_id=100)
        await manager.create("beta-bot", guild_id=GUILD_ID, channel_id=200)
        agents = await manager.list_agents(guild_id=GUILD_ID)
        names = sorted(a.name for a in agents)
        assert names == ["alpha-bot", "beta-bot"]


class TestAgentConfigure:
    async def test_configure_agent_updates_json(
        self, manager: AgentManager, tmp_chorus_home: Path
    ) -> None:
        await manager.create("test-bot", guild_id=GUILD_ID, channel_id=100)
        await manager.configure("test-bot", "system_prompt", "New prompt")

        # Verify agent.json was updated
        import json

        agent_json = tmp_chorus_home / "agents" / "test-bot" / "agent.json"
        data = json.loads(agent_json.read_text())
        assert data["system_prompt"] == "New prompt"

    async def test_configure_rejects_invalid_key(self, manager: AgentManager) -> None:
        await manager.create("test-bot", guild_id=GUILD_ID, channel_id=100)
        with pytest.raises(ValueError, match="Cannot configure"):
            await manager.configure("test-bot", "name", "new-name")

    async def test_configure_nonexistent_raises(self, manager: AgentManager) -> None:
        with pytest.raises(AgentNotFoundError):
            await manager.configure("nonexistent", "model", "gpt-4o")
