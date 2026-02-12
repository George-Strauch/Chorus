"""Tests for chorus.bot — Discord bot setup and event handling."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from discord import app_commands

from chorus.bot import ChorusBot
from chorus.config import BotConfig

if TYPE_CHECKING:
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
