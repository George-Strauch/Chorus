"""Container status slash command — /container-status."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

import discord
from discord import app_commands
from discord.ext import commands

logger = logging.getLogger("chorus.commands.container")


class ContainerStatusCog(commands.Cog):
    """Cog for the /container-status command."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(
        name="container-status",
        description="Show the container's git commit and uptime",
    )
    async def container_status(self, interaction: discord.Interaction) -> None:
        git_commit = os.environ.get("GIT_COMMIT", "unknown")
        start_time: datetime = getattr(self.bot, "start_time", datetime.now(UTC))
        now = datetime.now(UTC)
        delta = now - start_time

        total_seconds = int(delta.total_seconds())
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts: list[str] = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        parts.append(f"{minutes}m {seconds}s")
        uptime_str = " ".join(parts)

        embed = discord.Embed(
            title="Container Status",
            color=discord.Color.blurple(),
        )
        embed.add_field(name="Git Commit", value=f"`{git_commit}`", inline=True)
        embed.add_field(name="Uptime", value=uptime_str, inline=True)
        embed.add_field(
            name="Started",
            value=f"<t:{int(start_time.timestamp())}:R>",
            inline=True,
        )

        await interaction.response.send_message(embed=embed, ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    """Entry point for cog loading."""
    await bot.add_cog(ContainerStatusCog(bot))
