"""Tests for chorus.commands.model_commands — model shortcut slash commands."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import discord
import pytest
from discord.ext import commands

from chorus.commands.model_commands import _DISPLAY_NAMES, MODEL_SHORTCUTS, ModelCog


@pytest.fixture
def mock_bot() -> MagicMock:
    bot = MagicMock(spec=commands.Bot)
    bot.run_model_query = AsyncMock()
    return bot


@pytest.fixture
def cog(mock_bot: MagicMock) -> ModelCog:
    return ModelCog(mock_bot)


class TestModelShortcutsMappings:
    """Verify model shortcut dict has the expected entries."""

    def test_all_10_shortcuts_defined(self) -> None:
        assert len(MODEL_SHORTCUTS) == 10

    def test_display_names_match_shortcuts(self) -> None:
        assert set(_DISPLAY_NAMES.keys()) == set(MODEL_SHORTCUTS.keys())

    @pytest.mark.parametrize(
        "key,expected_model",
        [
            ("haiku", "claude-haiku-4-5-20251001"),
            ("sonnet", "claude-sonnet-4-5-20250929"),
            ("opus", "claude-opus-4-6"),
            ("gpt_4_1_nano", "gpt-4.1-nano"),
            ("gpt_5_nano", "gpt-5-nano"),
            ("gpt_5", "gpt-5"),
            ("gpt_5_pro", "gpt-5-pro"),
            ("gpt_5_2_pro", "gpt-5.2-pro"),
            ("o3_pro", "o3-pro"),
            ("o4_mini", "o4-mini"),
        ],
    )
    def test_shortcut_maps_to_model_id(self, key: str, expected_model: str) -> None:
        assert MODEL_SHORTCUTS[key] == expected_model


class TestModelCogCommands:
    """Verify cog has all expected commands."""

    def test_cog_has_10_app_commands(self, cog: ModelCog) -> None:
        cmd_names = {cmd.name for cmd in cog.__cog_app_commands__}
        expected = {
            "haiku", "sonnet", "opus",
            "gpt-4-1-nano", "gpt-5-nano", "gpt-5",
            "gpt-5-pro", "gpt-5-2-pro", "o3-pro", "o4-mini",
        }
        assert cmd_names == expected

    def test_all_commands_have_prompt_parameter(self, cog: ModelCog) -> None:
        for cmd in cog.__cog_app_commands__:
            param_names = [p.name for p in cmd.parameters]
            assert "prompt" in param_names, f"Command {cmd.name} missing 'prompt' parameter"


class TestRunModelShortcut:
    """Test the shared _run_model_shortcut handler."""

    def _make_interaction(self, *, channel_id: int = 100) -> MagicMock:
        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.defer = AsyncMock()
        interaction.followup = MagicMock()
        interaction.followup.send = AsyncMock()

        channel = MagicMock(spec=discord.TextChannel)
        channel.id = channel_id
        channel.send = AsyncMock()
        interaction.channel = channel

        user = MagicMock(spec=discord.Member)
        user.id = 42
        user.guild_permissions = MagicMock()
        user.guild_permissions.manage_guild = False
        interaction.user = user

        return interaction

    async def test_run_model_shortcut_calls_run_model_query(
        self, cog: ModelCog, mock_bot: MagicMock
    ) -> None:
        interaction = self._make_interaction()

        await cog._run_model_shortcut(interaction, "Hello!", "haiku")

        interaction.response.defer.assert_called_once()
        interaction.channel.send.assert_called_once_with("**[Haiku]** Hello!")
        mock_bot.run_model_query.assert_called_once_with(
            channel=interaction.channel,
            author_id=42,
            is_admin=False,
            prompt="Hello!",
            model_id="claude-haiku-4-5-20251001",
        )

    async def test_run_model_shortcut_reports_value_error(
        self, cog: ModelCog, mock_bot: MagicMock
    ) -> None:
        mock_bot.run_model_query.side_effect = ValueError("No agent bound to this channel.")
        interaction = self._make_interaction()

        await cog._run_model_shortcut(interaction, "test", "sonnet")

        interaction.followup.send.assert_called_once_with(
            "No agent bound to this channel.", ephemeral=True
        )

    async def test_run_model_shortcut_reports_generic_error(
        self, cog: ModelCog, mock_bot: MagicMock
    ) -> None:
        mock_bot.run_model_query.side_effect = RuntimeError("boom")
        interaction = self._make_interaction()

        await cog._run_model_shortcut(interaction, "test", "opus")

        interaction.followup.send.assert_called_once_with(
            "An error occurred while processing your request.", ephemeral=True
        )

    async def test_admin_user_passes_is_admin_true(
        self, cog: ModelCog, mock_bot: MagicMock
    ) -> None:
        interaction = self._make_interaction()
        interaction.user.guild_permissions.manage_guild = True

        await cog._run_model_shortcut(interaction, "test", "haiku")

        call_kwargs = mock_bot.run_model_query.call_args.kwargs
        assert call_kwargs["is_admin"] is True

    async def test_each_shortcut_uses_correct_model(
        self, cog: ModelCog, mock_bot: MagicMock
    ) -> None:
        """Verify each shortcut key maps to the right model_id in the call."""
        for key, model_id in MODEL_SHORTCUTS.items():
            mock_bot.run_model_query.reset_mock()
            interaction = self._make_interaction()

            await cog._run_model_shortcut(interaction, "test", key)

            call_kwargs = mock_bot.run_model_query.call_args.kwargs
            assert call_kwargs["model_id"] == model_id, (
                f"Shortcut {key} should use {model_id}"
            )


class TestModelCogSetup:
    """Test cog loading entry point."""

    async def test_setup_adds_cog(self) -> None:
        from chorus.commands.model_commands import setup

        bot = MagicMock(spec=commands.Bot)
        bot.add_cog = AsyncMock()

        await setup(bot)

        bot.add_cog.assert_called_once()
        added_cog = bot.add_cog.call_args.args[0]
        assert isinstance(added_cog, ModelCog)
