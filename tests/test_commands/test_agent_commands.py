"""Tests for chorus.commands.agent_commands — /agent slash commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from chorus.commands.agent_commands import AgentCog
from chorus.models import Agent, InvalidAgentNameError

if TYPE_CHECKING:
    from pathlib import Path


class TestPing:
    async def test_ping_command_returns_latency_message(self, mock_bot: MagicMock) -> None:
        mock_bot.latency = 0.045
        cog = AgentCog(mock_bot)

        interaction = MagicMock()
        interaction.response = AsyncMock()

        await cog.ping.callback(cog, interaction)

        interaction.response.send_message.assert_called_once()
        msg = interaction.response.send_message.call_args[0][0]
        assert "45" in msg
        assert "ms" in msg.lower()


class TestAgentInit:
    async def test_agent_init_creates_agent_and_channel(self, mock_bot: MagicMock) -> None:
        mock_agent = Agent(name="test-bot", channel_id=555)
        mock_bot.agent_manager = MagicMock()
        mock_bot.agent_manager.create = AsyncMock(return_value=mock_agent)

        # Mock guild with category
        mock_channel = MagicMock()
        mock_channel.id = 555
        mock_channel.mention = "<#555>"

        mock_category = MagicMock()
        mock_category.name = "Chorus Agents"

        interaction = MagicMock()
        interaction.response = AsyncMock()
        interaction.followup = AsyncMock()
        interaction.guild_id = 123
        interaction.guild = MagicMock()
        interaction.guild.categories = [mock_category]
        interaction.guild.create_text_channel = AsyncMock(return_value=mock_channel)

        cog = AgentCog(mock_bot)
        await cog.agent_init.callback(
            cog, interaction, name="test-bot", system_prompt=None, model=None, permissions=None
        )

        interaction.guild.create_text_channel.assert_called_once_with(
            "test-bot", category=mock_category
        )
        mock_bot.agent_manager.create.assert_called_once()
        interaction.response.defer.assert_called_once()
        interaction.followup.send.assert_called_once()
        msg = interaction.followup.send.call_args[0][0]
        assert "test-bot" in msg

    async def test_agent_init_validates_name(self, mock_bot: MagicMock) -> None:
        mock_bot.agent_manager = MagicMock()
        mock_bot.agent_manager.create = AsyncMock(
            side_effect=InvalidAgentNameError("Invalid name")
        )

        mock_channel = MagicMock()
        mock_channel.id = 555
        mock_channel.delete = AsyncMock()
        interaction = MagicMock()
        interaction.response = AsyncMock()
        interaction.followup = AsyncMock()
        interaction.guild_id = 123
        interaction.guild = MagicMock()
        interaction.guild.categories = []
        mock_category = MagicMock()
        interaction.guild.create_category = AsyncMock(return_value=mock_category)
        interaction.guild.create_text_channel = AsyncMock(return_value=mock_channel)

        cog = AgentCog(mock_bot)
        await cog.agent_init.callback(
            cog, interaction, name="BAD!", system_prompt=None, model=None, permissions=None
        )

        # Should send ephemeral error via followup
        interaction.followup.send.assert_called_once()
        call_kwargs = interaction.followup.send.call_args
        assert call_kwargs.kwargs.get("ephemeral") is True


class TestAgentDestroy:
    async def test_agent_destroy_sends_confirmation(self, mock_bot: MagicMock) -> None:
        mock_bot.agent_manager = MagicMock()

        interaction = MagicMock()
        interaction.response = AsyncMock()

        cog = AgentCog(mock_bot)
        await cog.agent_destroy.callback(cog, interaction, name="test-bot")

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args
        assert call_kwargs.kwargs.get("ephemeral") is True
        assert "view" in call_kwargs.kwargs


class TestAgentList:
    async def test_agent_list_returns_embed(self, mock_bot: MagicMock) -> None:
        agents = [
            Agent(name="alpha-bot", channel_id=100, permissions="standard"),
            Agent(name="beta-bot", channel_id=200, permissions="open"),
        ]
        mock_bot.agent_manager = MagicMock()
        mock_bot.agent_manager.list_agents = AsyncMock(return_value=agents)

        interaction = MagicMock()
        interaction.response = AsyncMock()
        interaction.guild_id = 123

        cog = AgentCog(mock_bot)
        await cog.agent_list.callback(cog, interaction)

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args
        assert call_kwargs.kwargs.get("embed") is not None


class TestAgentConfig:
    async def test_agent_config_updates_value(self, mock_bot: MagicMock) -> None:
        mock_bot.agent_manager = MagicMock()
        mock_bot.agent_manager.configure = AsyncMock()

        interaction = MagicMock()
        interaction.response = AsyncMock()

        cog = AgentCog(mock_bot)
        await cog.agent_config.callback(
            cog, interaction, name="test-bot", key="system_prompt", value="New prompt"
        )

        mock_bot.agent_manager.configure.assert_called_once_with(
            "test-bot", "system_prompt", "New prompt"
        )
        interaction.response.send_message.assert_called_once()
        msg = interaction.response.send_message.call_args[0][0]
        assert "test-bot" in msg


def _write_model_cache(chorus_home: Path, models: list[str]) -> None:
    """Write a fake available_models.json cache."""
    cache = {
        "last_updated": "2026-02-12T10:00:00Z",
        "providers": {
            "anthropic": {"valid": True, "models": models},
        },
    }
    (chorus_home / "available_models.json").write_text(json.dumps(cache))


class TestConfigKeyAutocomplete:
    @pytest.mark.asyncio
    async def test_key_autocomplete_returns_configurable_keys(self, mock_bot: MagicMock) -> None:
        interaction = MagicMock()
        cog = AgentCog(mock_bot)
        results = await cog._config_key_autocomplete(interaction, "")
        names = [r.name for r in results]
        assert "model" in names
        assert "permissions" in names
        assert "system_prompt" in names

    @pytest.mark.asyncio
    async def test_key_autocomplete_filters(self, mock_bot: MagicMock) -> None:
        interaction = MagicMock()
        cog = AgentCog(mock_bot)
        results = await cog._config_key_autocomplete(interaction, "mod")
        names = [r.name for r in results]
        assert "model" in names
        assert "permissions" not in names


class TestConfigValueAutocomplete:
    @pytest.mark.asyncio
    async def test_value_autocomplete_model_returns_models(
        self, mock_bot: MagicMock, tmp_chorus_home: Path
    ) -> None:
        _write_model_cache(tmp_chorus_home, ["claude-sonnet-4-20250514", "gpt-4o"])
        interaction = MagicMock()
        interaction.namespace = MagicMock()
        interaction.namespace.key = "model"
        cog = AgentCog(mock_bot)
        results = await cog._config_value_autocomplete(interaction, "")
        names = [r.name for r in results]
        assert "claude-sonnet-4-20250514" in names
        assert "gpt-4o" in names

    @pytest.mark.asyncio
    async def test_value_autocomplete_permissions_returns_presets(
        self, mock_bot: MagicMock
    ) -> None:
        interaction = MagicMock()
        interaction.namespace = MagicMock()
        interaction.namespace.key = "permissions"
        cog = AgentCog(mock_bot)
        results = await cog._config_value_autocomplete(interaction, "")
        names = [r.name for r in results]
        assert "open" in names
        assert "standard" in names
        assert "locked" in names

    @pytest.mark.asyncio
    async def test_value_autocomplete_other_key_returns_empty(
        self, mock_bot: MagicMock
    ) -> None:
        interaction = MagicMock()
        interaction.namespace = MagicMock()
        interaction.namespace.key = "system_prompt"
        cog = AgentCog(mock_bot)
        results = await cog._config_value_autocomplete(interaction, "")
        assert results == []


class TestInitModelAutocomplete:
    @pytest.mark.asyncio
    async def test_init_model_autocomplete(
        self, mock_bot: MagicMock, tmp_chorus_home: Path
    ) -> None:
        _write_model_cache(tmp_chorus_home, ["claude-sonnet-4-20250514", "gpt-4o"])
        interaction = MagicMock()
        cog = AgentCog(mock_bot)
        results = await cog._init_model_autocomplete(interaction, "claude")
        names = [r.name for r in results]
        assert "claude-sonnet-4-20250514" in names
        assert "gpt-4o" not in names


class TestInitPermissionsAutocomplete:
    @pytest.mark.asyncio
    async def test_init_permissions_autocomplete(self, mock_bot: MagicMock) -> None:
        interaction = MagicMock()
        cog = AgentCog(mock_bot)
        results = await cog._init_permissions_autocomplete(interaction, "")
        names = [r.name for r in results]
        assert "open" in names
        assert "standard" in names
        assert "locked" in names
