"""Tests for chorus.commands.container_commands — /container-status."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest


@pytest.fixture
def container_bot() -> MagicMock:
    """Mock bot with start_time set."""
    bot = MagicMock()
    bot.start_time = datetime(2026, 2, 15, 10, 0, 0, tzinfo=UTC)
    return bot


@pytest.fixture
def container_interaction() -> MagicMock:
    """Mock interaction."""
    interaction = MagicMock(spec=discord.Interaction)
    interaction.response = AsyncMock()
    return interaction


class TestContainerStatus:
    @pytest.mark.asyncio
    async def test_shows_git_commit_from_env(
        self, container_bot: MagicMock, container_interaction: MagicMock
    ) -> None:
        from chorus.commands.container_commands import ContainerStatusCog

        cog = ContainerStatusCog(container_bot)
        with patch.dict("os.environ", {"GIT_COMMIT": "abc1234"}):
            await cog.container_status.callback(cog, container_interaction)
        embed = container_interaction.response.send_message.call_args.kwargs["embed"]
        fields = {f.name: f.value for f in embed.fields}
        assert fields["Git Commit"] == "`abc1234`"

    @pytest.mark.asyncio
    async def test_shows_unknown_when_no_env(
        self, container_bot: MagicMock, container_interaction: MagicMock
    ) -> None:
        from chorus.commands.container_commands import ContainerStatusCog

        cog = ContainerStatusCog(container_bot)
        with patch.dict("os.environ", {}, clear=False):
            # Remove GIT_COMMIT if present
            import os

            env = os.environ.copy()
            env.pop("GIT_COMMIT", None)
            with patch.dict("os.environ", env, clear=True):
                await cog.container_status.callback(cog, container_interaction)
        embed = container_interaction.response.send_message.call_args.kwargs["embed"]
        fields = {f.name: f.value for f in embed.fields}
        assert fields["Git Commit"] == "`unknown`"

    @pytest.mark.asyncio
    async def test_shows_uptime(
        self, container_bot: MagicMock, container_interaction: MagicMock
    ) -> None:
        from chorus.commands.container_commands import ContainerStatusCog

        # Bot started 2h 30m 15s ago
        now = datetime(2026, 2, 15, 12, 30, 15, tzinfo=UTC)
        container_bot.start_time = now - timedelta(hours=2, minutes=30, seconds=15)
        cog = ContainerStatusCog(container_bot)
        with patch("chorus.commands.container_commands.datetime") as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            await cog.container_status.callback(cog, container_interaction)
        embed = container_interaction.response.send_message.call_args.kwargs["embed"]
        fields = {f.name: f.value for f in embed.fields}
        assert "2h" in fields["Uptime"]
        assert "30m" in fields["Uptime"]
        assert "15s" in fields["Uptime"]

    @pytest.mark.asyncio
    async def test_shows_days_in_uptime(
        self, container_bot: MagicMock, container_interaction: MagicMock
    ) -> None:
        from chorus.commands.container_commands import ContainerStatusCog

        now = datetime(2026, 2, 18, 10, 0, 0, tzinfo=UTC)
        container_bot.start_time = now - timedelta(days=3, hours=5)
        cog = ContainerStatusCog(container_bot)
        with patch("chorus.commands.container_commands.datetime") as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            await cog.container_status.callback(cog, container_interaction)
        embed = container_interaction.response.send_message.call_args.kwargs["embed"]
        fields = {f.name: f.value for f in embed.fields}
        assert "3d" in fields["Uptime"]
        assert "5h" in fields["Uptime"]

    @pytest.mark.asyncio
    async def test_is_ephemeral(
        self, container_bot: MagicMock, container_interaction: MagicMock
    ) -> None:
        from chorus.commands.container_commands import ContainerStatusCog

        cog = ContainerStatusCog(container_bot)
        await cog.container_status.callback(cog, container_interaction)
        assert (
            container_interaction.response.send_message.call_args.kwargs.get("ephemeral")
            is True
        )

    @pytest.mark.asyncio
    async def test_embed_has_started_field(
        self, container_bot: MagicMock, container_interaction: MagicMock
    ) -> None:
        from chorus.commands.container_commands import ContainerStatusCog

        cog = ContainerStatusCog(container_bot)
        await cog.container_status.callback(cog, container_interaction)
        embed = container_interaction.response.send_message.call_args.kwargs["embed"]
        fields = {f.name: f.value for f in embed.fields}
        assert "Started" in fields
        # Discord relative timestamp format
        assert "<t:" in fields["Started"]

    @pytest.mark.asyncio
    async def test_setup_adds_cog(self) -> None:
        from chorus.commands.container_commands import setup

        bot = MagicMock()
        bot.add_cog = AsyncMock()
        await setup(bot)
        bot.add_cog.assert_called_once()
