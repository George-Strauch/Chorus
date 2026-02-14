"""Tests for chorus.commands.info_commands — /models and /permissions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from chorus.config import GlobalConfig

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def info_bot(tmp_chorus_home: Path) -> MagicMock:
    """Mock bot with GlobalConfig and chorus_home."""
    from chorus.config import BotConfig

    config = BotConfig(
        discord_token="test-token",
        chorus_home=tmp_chorus_home,
    )
    bot = MagicMock()
    bot.config = config
    bot.global_config = GlobalConfig.load(tmp_chorus_home / "config.json")
    return bot


@pytest.fixture
def info_interaction() -> MagicMock:
    """Mock interaction."""
    interaction = MagicMock(spec=discord.Interaction)
    interaction.response = AsyncMock()
    return interaction


def _write_model_cache(chorus_home: Path, models: list[str]) -> None:
    cache = {
        "last_updated": "2026-02-12T10:00:00Z",
        "providers": {
            "anthropic": {"valid": True, "models": models},
        },
    }
    (chorus_home / "available_models.json").write_text(json.dumps(cache))


class TestModelsCommand:
    @pytest.mark.asyncio
    async def test_models_lists_available_models(
        self, info_bot: MagicMock, info_interaction: MagicMock, tmp_chorus_home: Path
    ) -> None:
        from chorus.commands.info_commands import InfoCog

        _write_model_cache(tmp_chorus_home, ["claude-sonnet-4-20250514", "gpt-4o"])
        cog = InfoCog(info_bot)
        await cog.models_command.callback(cog, info_interaction)
        embed = info_interaction.response.send_message.call_args.kwargs["embed"]
        embed_text = str(embed.to_dict())
        assert "claude-sonnet-4-20250514" in embed_text
        assert "gpt-4o" in embed_text

    @pytest.mark.asyncio
    async def test_models_no_cache_shows_message(
        self, info_bot: MagicMock, info_interaction: MagicMock
    ) -> None:
        from chorus.commands.info_commands import InfoCog

        cog = InfoCog(info_bot)
        await cog.models_command.callback(cog, info_interaction)
        msg = str(info_interaction.response.send_message.call_args)
        assert "validate-keys" in msg

    @pytest.mark.asyncio
    async def test_models_highlights_default(
        self, info_bot: MagicMock, info_interaction: MagicMock, tmp_chorus_home: Path
    ) -> None:
        from chorus.commands.info_commands import InfoCog

        _write_model_cache(tmp_chorus_home, ["claude-sonnet-4-20250514", "gpt-4o"])
        info_bot.global_config.default_model = "claude-sonnet-4-20250514"
        cog = InfoCog(info_bot)
        await cog.models_command.callback(cog, info_interaction)
        embed = info_interaction.response.send_message.call_args.kwargs["embed"]
        embed_text = str(embed.to_dict())
        assert "**>**" in embed_text
        assert "Current default: claude-sonnet-4-20250514" in embed_text

    @pytest.mark.asyncio
    async def test_models_is_ephemeral(
        self, info_bot: MagicMock, info_interaction: MagicMock, tmp_chorus_home: Path
    ) -> None:
        from chorus.commands.info_commands import InfoCog

        _write_model_cache(tmp_chorus_home, ["claude-sonnet-4-20250514"])
        cog = InfoCog(info_bot)
        await cog.models_command.callback(cog, info_interaction)
        assert info_interaction.response.send_message.call_args.kwargs.get(
            "ephemeral"
        ) is True

    @pytest.mark.asyncio
    async def test_models_shows_invalid_key_provider(
        self, info_bot: MagicMock, info_interaction: MagicMock, tmp_chorus_home: Path
    ) -> None:
        from chorus.commands.info_commands import InfoCog

        cache = {
            "last_updated": "2026-02-12T10:00:00Z",
            "providers": {
                "openai": {"valid": False, "models": []},
            },
        }
        (tmp_chorus_home / "available_models.json").write_text(json.dumps(cache))
        cog = InfoCog(info_bot)
        await cog.models_command.callback(cog, info_interaction)
        embed = info_interaction.response.send_message.call_args.kwargs["embed"]
        embed_text = str(embed.to_dict())
        assert "Invalid API key" in embed_text


class TestPermissionsCommand:
    @pytest.mark.asyncio
    async def test_permissions_lists_all_presets(
        self, info_bot: MagicMock, info_interaction: MagicMock
    ) -> None:
        from chorus.commands.info_commands import InfoCog

        cog = InfoCog(info_bot)
        await cog.permissions_command.callback(cog, info_interaction)
        embed = info_interaction.response.send_message.call_args.kwargs["embed"]
        embed_text = str(embed.to_dict())
        assert "open" in embed_text
        assert "standard" in embed_text
        assert "locked" in embed_text

    @pytest.mark.asyncio
    async def test_permissions_marks_default(
        self, info_bot: MagicMock, info_interaction: MagicMock
    ) -> None:
        from chorus.commands.info_commands import InfoCog

        info_bot.global_config.default_permissions = "open"
        cog = InfoCog(info_bot)
        await cog.permissions_command.callback(cog, info_interaction)
        embed = info_interaction.response.send_message.call_args.kwargs["embed"]
        embed_text = str(embed.to_dict())
        assert "(default)" in embed_text
        assert "Current default: open" in embed_text

    @pytest.mark.asyncio
    async def test_permissions_shows_allow_ask_rules(
        self, info_bot: MagicMock, info_interaction: MagicMock
    ) -> None:
        from chorus.commands.info_commands import InfoCog

        cog = InfoCog(info_bot)
        await cog.permissions_command.callback(cog, info_interaction)
        embed = info_interaction.response.send_message.call_args.kwargs["embed"]
        embed_text = str(embed.to_dict())
        assert "Allow" in embed_text
        assert "Ask" in embed_text

    @pytest.mark.asyncio
    async def test_permissions_is_ephemeral(
        self, info_bot: MagicMock, info_interaction: MagicMock
    ) -> None:
        from chorus.commands.info_commands import InfoCog

        cog = InfoCog(info_bot)
        await cog.permissions_command.callback(cog, info_interaction)
        assert info_interaction.response.send_message.call_args.kwargs.get(
            "ephemeral"
        ) is True
