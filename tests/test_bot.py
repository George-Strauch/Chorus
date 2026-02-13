"""Tests for chorus.bot — Discord bot setup and event handling."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from discord import app_commands

from chorus.agent.directory import AgentDirectory
from chorus.agent.manager import AgentManager
from chorus.bot import ChorusBot
from chorus.config import BotConfig
from chorus.storage.db import Database

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


@pytest.fixture
def bot_config(tmp_chorus_home: Path) -> BotConfig:
    return BotConfig(
        discord_token="test-token",
        chorus_home=tmp_chorus_home,
    )


@pytest.fixture
def bot(bot_config: BotConfig) -> ChorusBot:
    return ChorusBot(bot_config)


@pytest.fixture
async def reconcile_db(tmp_chorus_home: Path) -> AsyncGenerator[Database, None]:
    """Database for reconciliation tests."""
    db = Database(tmp_chorus_home / "db" / "chorus.db")
    await db.init()
    yield db
    await db.close()


class TestBotSetup:
    def test_bot_creates_with_correct_intents(self, bot: ChorusBot) -> None:
        intents = bot.intents
        assert intents.guilds is True
        assert intents.guild_messages is True
        assert intents.message_content is True
        assert intents.presences is False
        assert intents.members is False

    async def test_bot_loads_cogs_from_commands_package(self, bot: ChorusBot) -> None:
        with patch.object(bot, "load_extension", new_callable=AsyncMock) as mock_load:
            await bot.setup_hook()
            # Should load at least agent_commands
            loaded = [call.args[0] for call in mock_load.call_args_list]
            assert "chorus.commands.agent_commands" in loaded


class TestBotEvents:
    async def test_bot_on_message_ignores_self(self, bot: ChorusBot) -> None:
        # Set up bot.user
        bot._connection.user = MagicMock(spec=discord.User)
        bot._connection.user.id = 444555666

        message = MagicMock(spec=discord.Message)
        message.author = bot._connection.user

        with patch.object(bot, "process_commands", new_callable=AsyncMock) as mock_process:
            await bot.on_message(message)
            mock_process.assert_not_called()

    async def test_bot_on_message_processes_others(self, bot: ChorusBot) -> None:
        bot._connection.user = MagicMock(spec=discord.User)
        bot._connection.user.id = 444555666

        message = MagicMock(spec=discord.Message)
        other_user = MagicMock(spec=discord.User)
        other_user.id = 999999
        message.author = other_user

        with patch.object(bot, "process_commands", new_callable=AsyncMock) as mock_process:
            await bot.on_message(message)
            mock_process.assert_called_once_with(message)


class TestBotErrorHandler:
    async def test_bot_error_handler_sends_ephemeral(self, bot: ChorusBot) -> None:
        with patch.object(bot, "load_extension", new_callable=AsyncMock):
            await bot.setup_hook()

        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.is_done.return_value = False
        interaction.response.send_message = AsyncMock()
        interaction.followup = AsyncMock()

        error = app_commands.MissingPermissions(["admin"])

        await bot.tree.on_error(interaction, error)
        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args
        assert call_kwargs.kwargs.get("ephemeral") is True


class TestReconcileChannels:
    """Tests for _reconcile_channels() startup reconciliation."""

    def _make_mock_channel(self, channel_id: int, name: str) -> MagicMock:
        ch = MagicMock(spec=discord.TextChannel)
        ch.id = channel_id
        ch.name = name
        ch.delete = AsyncMock()
        return ch

    def _make_bot_with_guild(
        self,
        bot_config: BotConfig,
        guild_id: int,
        category_channels: list[MagicMock],
        all_text_channels: list[MagicMock] | None = None,
    ) -> ChorusBot:
        config = BotConfig(
            discord_token=bot_config.discord_token,
            chorus_home=bot_config.chorus_home,
            dev_guild_id=guild_id,
        )
        bot = ChorusBot(config)

        # Mock category
        category = MagicMock()
        category.name = "Chorus Agents"
        category.text_channels = category_channels

        # Mock guild
        guild = MagicMock(spec=discord.Guild)
        guild.id = guild_id
        guild.categories = [category]
        all_channels = all_text_channels if all_text_channels is not None else category_channels
        guild.text_channels = all_channels
        guild.create_text_channel = AsyncMock()

        bot.get_guild = MagicMock(return_value=guild)
        return bot

    async def test_reconcile_skips_without_dev_guild(
        self, bot_config: BotConfig, reconcile_db: Database
    ) -> None:
        """No dev_guild_id → no-op."""
        bot = ChorusBot(bot_config)  # no dev_guild_id
        bot.db = reconcile_db
        # Should return without error
        await bot._reconcile_channels()

    async def test_reconcile_skips_without_category(
        self, bot_config: BotConfig, reconcile_db: Database
    ) -> None:
        """No 'Chorus Agents' category → no-op."""
        config = BotConfig(
            discord_token="test-token",
            chorus_home=bot_config.chorus_home,
            dev_guild_id=123,
        )
        bot = ChorusBot(config)
        bot.db = reconcile_db

        guild = MagicMock(spec=discord.Guild)
        guild.id = 123
        guild.categories = []  # No categories
        bot.get_guild = MagicMock(return_value=guild)

        await bot._reconcile_channels()
        # No error = pass

    async def test_reconcile_deletes_ghost_channels(
        self, bot_config: BotConfig, reconcile_db: Database
    ) -> None:
        """Channel in category with no DB agent → deleted."""
        ghost_channel = self._make_mock_channel(111, "ghost-agent")
        bot = self._make_bot_with_guild(bot_config, 999, [ghost_channel])
        bot.db = reconcile_db

        await bot._reconcile_channels()

        ghost_channel.delete.assert_called_once_with(
            reason="Chorus reconciliation: no agent record"
        )

    async def test_reconcile_recreates_missing_channels(
        self,
        bot_config: BotConfig,
        reconcile_db: Database,
        tmp_chorus_home: Path,
        tmp_template: Path,
    ) -> None:
        """DB agent with no Discord channel → channel created, IDs updated."""
        # Register an agent in DB pointing to a channel that doesn't exist
        await reconcile_db.register_agent(
            name="orphan-agent",
            channel_id=555,
            guild_id=999,
            model=None,
            permissions="standard",
            created_at="2026-01-01T00:00:00",
        )

        # Create the agent directory so update_channel_id works
        directory = AgentDirectory(tmp_chorus_home, tmp_template)
        directory.create("orphan-agent")

        # No channels exist in Discord (empty list for category + guild)
        bot = self._make_bot_with_guild(bot_config, 999, [], [])
        bot.db = reconcile_db
        bot.agent_manager = AgentManager(directory, reconcile_db)

        # Mock the new channel that will be created
        new_channel = self._make_mock_channel(777, "orphan-agent")
        guild = bot.get_guild(999)
        guild.create_text_channel = AsyncMock(return_value=new_channel)

        await bot._reconcile_channels()

        guild.create_text_channel.assert_called_once()
        # DB should have the new channel_id
        agent = await reconcile_db.get_agent("orphan-agent")
        assert agent is not None
        assert agent["channel_id"] == 777

    async def test_reconcile_no_changes_when_synced(
        self, bot_config: BotConfig, reconcile_db: Database
    ) -> None:
        """Everything matches → nothing happens."""
        # Register an agent in DB
        await reconcile_db.register_agent(
            name="synced-agent",
            channel_id=222,
            guild_id=999,
            model=None,
            permissions="standard",
            created_at="2026-01-01T00:00:00",
        )

        # Channel exists in Discord with matching ID
        synced_channel = self._make_mock_channel(222, "synced-agent")
        bot = self._make_bot_with_guild(bot_config, 999, [synced_channel], [synced_channel])
        bot.db = reconcile_db

        await bot._reconcile_channels()

        # No deletions, no creations
        synced_channel.delete.assert_not_called()
        guild = bot.get_guild(999)
        guild.create_text_channel.assert_not_called()
