"""Tests for chorus.agent.manager — agent lifecycle."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from chorus.agent.directory import AgentDirectory
from chorus.agent.manager import AgentManager
from chorus.config import BotConfig
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


class TestAgentInitRefinement:
    """Integration tests for system prompt refinement during agent creation."""

    @pytest.fixture
    def bot_config(self, tmp_chorus_home: Path) -> BotConfig:
        return BotConfig(
            discord_token="test-token",
            anthropic_api_key="sk-ant-test",
            chorus_home=tmp_chorus_home,
        )

    @pytest.fixture
    def manager_with_config(
        self, agent_directory: AgentDirectory, db: Database, bot_config: BotConfig
    ) -> AgentManager:
        return AgentManager(agent_directory, db, bot_config=bot_config)

    async def test_agent_init_calls_refinement(
        self, manager_with_config: AgentManager
    ) -> None:
        """Verify refine_system_prompt is called during create()."""
        with patch(
            "chorus.agent.prompt_refinement.refine_system_prompt",
            new_callable=AsyncMock,
            return_value="Refined prompt for test-bot",
        ) as mock_refine:
            agent = await manager_with_config.create(
                "test-bot", guild_id=GUILD_ID, channel_id=100
            )
            mock_refine.assert_called_once()
            assert agent.system_prompt == "Refined prompt for test-bot"

    async def test_agent_init_stores_refined_prompt_in_agent_json(
        self, manager_with_config: AgentManager, tmp_chorus_home: Path
    ) -> None:
        """Verify refined prompt is written to the filesystem."""
        with patch(
            "chorus.agent.prompt_refinement.refine_system_prompt",
            new_callable=AsyncMock,
            return_value="You are a specialized documentation agent.",
        ):
            await manager_with_config.create(
                "docs-bot", guild_id=GUILD_ID, channel_id=100
            )

        agent_json = tmp_chorus_home / "agents" / "docs-bot" / "agent.json"
        data = json.loads(agent_json.read_text())
        assert data["system_prompt"] == "You are a specialized documentation agent."

    async def test_agent_init_succeeds_even_if_refinement_fails(
        self, manager_with_config: AgentManager
    ) -> None:
        """If refine_system_prompt raises, agent is still created with template prompt."""
        with patch(
            "chorus.agent.prompt_refinement.refine_system_prompt",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM exploded"),
        ):
            agent = await manager_with_config.create(
                "test-bot", guild_id=GUILD_ID, channel_id=100
            )
        # Agent created successfully with template prompt
        assert agent.name == "test-bot"
        assert "general-purpose" in agent.system_prompt.lower()

    async def test_agent_init_passes_user_description(
        self, manager_with_config: AgentManager
    ) -> None:
        """User-provided system_prompt override is passed as user_description."""
        with patch(
            "chorus.agent.prompt_refinement.refine_system_prompt",
            new_callable=AsyncMock,
            return_value="Refined with user input",
        ) as mock_refine:
            await manager_with_config.create(
                "code-bot",
                guild_id=GUILD_ID,
                channel_id=100,
                overrides={"system_prompt": "A Python coding assistant"},
            )
            call_kwargs = mock_refine.call_args
            assert call_kwargs[1]["user_description"] == "A Python coding assistant"

    async def test_agent_init_uses_user_override_on_refinement_failure(
        self, manager_with_config: AgentManager
    ) -> None:
        """If refinement fails and user provided a description, use it as-is."""
        with patch(
            "chorus.agent.prompt_refinement.refine_system_prompt",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            agent = await manager_with_config.create(
                "test-bot",
                guild_id=GUILD_ID,
                channel_id=100,
                overrides={"system_prompt": "My custom prompt"},
            )
        assert agent.system_prompt == "My custom prompt"
