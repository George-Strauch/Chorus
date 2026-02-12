"""Agent slash commands — /ping and future /agent group."""

from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

logger = logging.getLogger("chorus.commands.agent")


class AgentCog(commands.Cog):
    """Cog for agent management commands."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="ping", description="Check bot latency")
    async def ping(self, interaction: discord.Interaction) -> None:
        latency_ms = round(self.bot.latency * 1000, 1)
        await interaction.response.send_message(f"Pong! {latency_ms}ms")


async def setup(bot: commands.Bot) -> None:
    """Entry point for cog loading."""
    await bot.add_cog(AgentCog(bot))
