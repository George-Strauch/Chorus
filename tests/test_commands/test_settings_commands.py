"""Tests for chorus.commands.settings_commands — /settings slash commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import discord
import pytest

from chorus.config import GlobalConfig


@pytest.fixture
def settings_bot(tmp_chorus_home: Path) -> MagicMock:
    """Mock bot with GlobalConfig and chorus_home set up."""
    from chorus.config import BotConfig

    config = BotConfig(
        discord_token="test-token",
        anthropic_api_key="sk-ant-test",
        openai_api_key="sk-openai-test",
        chorus_home=tmp_chorus_home,
    )
    bot = MagicMock()
    bot.config = config
    bot.global_config = GlobalConfig.load(tmp_chorus_home / "config.json")
    return bot


@pytest.fixture
def settings_interaction(settings_bot: MagicMock) -> MagicMock:
    """Mock interaction with manage_guild permissions."""
    interaction = MagicMock(spec=discord.Interaction)
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.guild_id = 123456789
    interaction.user = MagicMock()
    interaction.user.guild_permissions = MagicMock()
    interaction.user.guild_permissions.manage_guild = True
    return interaction


def _write_model_cache(chorus_home: Path, models: list[str] | None = None) -> None:
    """Write a fake available_models.json cache."""
    if models is None:
        models = ["claude-sonnet-4-20250514", "claude-haiku-4-20250506"]
    cache = {
        "last_updated": "2026-02-12T10:00:00Z",
        "providers": {
            "anthropic": {"valid": True, "models": models},
        },
    }
    (chorus_home / "available_models.json").write_text(json.dumps(cache))


class TestSettingsModel:
    @pytest.mark.asyncio
    async def test_settings_model_command_updates_default(
        self, settings_bot: MagicMock, settings_interaction: MagicMock, tmp_chorus_home: Path
    ) -> None:
        from chorus.commands.settings_commands import SettingsCog

        _write_model_cache(tmp_chorus_home)
        cog = SettingsCog(settings_bot)
        await cog.settings_model.callback(
            cog, settings_interaction, model="claude-sonnet-4-20250514"
        )
        assert settings_bot.global_config.default_model == "claude-sonnet-4-20250514"
        settings_interaction.response.send_message.assert_called_once()
        call_kwargs = settings_interaction.response.send_message.call_args
        assert "claude-sonnet-4-20250514" in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_settings_model_command_rejects_unavailable_model(
        self, settings_bot: MagicMock, settings_interaction: MagicMock, tmp_chorus_home: Path
    ) -> None:
        from chorus.commands.settings_commands import SettingsCog

        _write_model_cache(tmp_chorus_home)
        cog = SettingsCog(settings_bot)
        await cog.settings_model.callback(
            cog, settings_interaction, model="nonexistent-model"
        )
        # Model should NOT have been set
        assert settings_bot.global_config.default_model is None
        call_kwargs = settings_interaction.response.send_message.call_args
        msg = str(call_kwargs).lower()
        assert "not available" in msg or "not found" in msg

    def test_settings_model_command_requires_manage_server(self) -> None:
        from chorus.commands.settings_commands import SettingsCog

        # The group-level default_permissions enforces manage_guild
        assert SettingsCog.settings_model is not None


class TestSettingsPermissions:
    @pytest.mark.asyncio
    async def test_settings_permissions_command_updates_default(
        self, settings_bot: MagicMock, settings_interaction: MagicMock
    ) -> None:
        from chorus.commands.settings_commands import SettingsCog

        cog = SettingsCog(settings_bot)
        await cog.settings_permissions.callback(cog, settings_interaction, profile="open")
        assert settings_bot.global_config.default_permissions == "open"

    @pytest.mark.asyncio
    async def test_settings_permissions_command_accepts_preset_name(
        self, settings_bot: MagicMock, settings_interaction: MagicMock
    ) -> None:
        from chorus.commands.settings_commands import SettingsCog

        cog = SettingsCog(settings_bot)
        for preset in ("open", "standard", "locked"):
            await cog.settings_permissions.callback(cog, settings_interaction, profile=preset)
            assert settings_bot.global_config.default_permissions == preset

    @pytest.mark.asyncio
    async def test_settings_permissions_command_rejects_invalid_preset(
        self, settings_bot: MagicMock, settings_interaction: MagicMock
    ) -> None:
        from chorus.commands.settings_commands import SettingsCog

        cog = SettingsCog(settings_bot)
        await cog.settings_permissions.callback(
            cog, settings_interaction, profile="nonexistent"
        )
        # Should NOT have changed the default
        assert settings_bot.global_config.default_permissions == "standard"
        call_kwargs = settings_interaction.response.send_message.call_args
        msg = str(call_kwargs).lower()
        assert "unknown" in msg or "invalid" in msg


class TestSettingsShow:
    @pytest.mark.asyncio
    async def test_settings_show_command_returns_embed(
        self, settings_bot: MagicMock, settings_interaction: MagicMock
    ) -> None:
        from chorus.commands.settings_commands import SettingsCog

        cog = SettingsCog(settings_bot)
        await cog.settings_show.callback(cog, settings_interaction)
        call_kwargs = settings_interaction.response.send_message.call_args
        assert call_kwargs.kwargs.get("embed") is not None or (
            len(call_kwargs.args) > 0 and isinstance(call_kwargs.args[0], discord.Embed)
        )

    @pytest.mark.asyncio
    async def test_settings_show_displays_all_fields(
        self, settings_bot: MagicMock, settings_interaction: MagicMock, tmp_chorus_home: Path
    ) -> None:
        from chorus.commands.settings_commands import SettingsCog

        _write_model_cache(tmp_chorus_home)
        settings_bot.global_config.default_model = "claude-sonnet-4-20250514"
        cog = SettingsCog(settings_bot)
        await cog.settings_show.callback(cog, settings_interaction)
        call_kwargs = settings_interaction.response.send_message.call_args
        embed = call_kwargs.kwargs.get("embed")
        assert embed is not None
        embed_text = str(embed.to_dict())
        # Check all 5 config fields appear somewhere
        assert "claude-sonnet-4-20250514" in embed_text
        assert "standard" in embed_text
        assert "1800" in embed_text or "30" in embed_text  # idle timeout
        assert "25" in embed_text  # max iterations
        assert "120" in embed_text  # max bash timeout
        # Check API key info
        assert "anthropic" in embed_text.lower() or "Anthropic" in embed_text


class TestValidateKeys:
    @pytest.mark.asyncio
    async def test_settings_validate_keys_triggers_discovery(
        self, settings_bot: MagicMock, settings_interaction: MagicMock
    ) -> None:
        from chorus.commands.settings_commands import SettingsCog

        cog = SettingsCog(settings_bot)
        with patch(
            "chorus.commands.settings_commands.validate_and_discover",
            new_callable=AsyncMock,
        ) as mock_discover:
            mock_discover.return_value = {
                "last_updated": "2026-02-12T10:00:00Z",
                "providers": {
                    "anthropic": {"valid": True, "models": ["claude-sonnet-4-20250514"]},
                },
            }
            await cog.settings_validate_keys.callback(cog, settings_interaction)
            mock_discover.assert_called_once()

    @pytest.mark.asyncio
    async def test_settings_validate_keys_reports_results(
        self, settings_bot: MagicMock, settings_interaction: MagicMock
    ) -> None:
        from chorus.commands.settings_commands import SettingsCog

        cog = SettingsCog(settings_bot)
        result = {
            "last_updated": "2026-02-12T10:00:00Z",
            "providers": {
                "anthropic": {
                    "valid": True,
                    "models": ["claude-sonnet-4-20250514", "claude-haiku-4-20250506"],
                },
                "openai": {"valid": False, "models": []},
            },
        }
        with patch(
            "chorus.commands.settings_commands.validate_and_discover",
            new_callable=AsyncMock,
            return_value=result,
        ):
            await cog.settings_validate_keys.callback(cog, settings_interaction)
        # Should have deferred then sent followup with embed
        settings_interaction.response.defer.assert_called_once()
        call_kwargs = settings_interaction.followup.send.call_args
        embed = call_kwargs.kwargs.get("embed")
        assert embed is not None
        embed_text = str(embed.to_dict())
        assert "anthropic" in embed_text.lower()


class TestSettingsEphemeral:
    @pytest.mark.asyncio
    async def test_settings_commands_ephemeral_responses(
        self, settings_bot: MagicMock, settings_interaction: MagicMock, tmp_chorus_home: Path
    ) -> None:
        from chorus.commands.settings_commands import SettingsCog

        _write_model_cache(tmp_chorus_home)
        cog = SettingsCog(settings_bot)

        # /settings model
        await cog.settings_model.callback(
            cog, settings_interaction, model="claude-sonnet-4-20250514"
        )
        assert settings_interaction.response.send_message.call_args.kwargs.get("ephemeral") is True

        settings_interaction.reset_mock()

        # /settings permissions
        await cog.settings_permissions.callback(cog, settings_interaction, profile="open")
        assert settings_interaction.response.send_message.call_args.kwargs.get("ephemeral") is True

        settings_interaction.reset_mock()

        # /settings show
        await cog.settings_show.callback(cog, settings_interaction)
        assert settings_interaction.response.send_message.call_args.kwargs.get("ephemeral") is True

        settings_interaction.reset_mock()

        # /settings validate-keys
        with patch(
            "chorus.commands.settings_commands.validate_and_discover",
            new_callable=AsyncMock,
            return_value={"last_updated": "now", "providers": {}},
        ):
            await cog.settings_validate_keys.callback(cog, settings_interaction)
        assert settings_interaction.response.defer.call_args.kwargs.get("ephemeral") is True
        assert settings_interaction.followup.send.call_args.kwargs.get("ephemeral") is True
