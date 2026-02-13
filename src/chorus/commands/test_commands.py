"""Live test slash commands — /test run, /test list, /test status."""

from __future__ import annotations

import logging
import time

import discord
from discord import app_commands
from discord.ext import commands

from chorus.testing.runner import SuiteResult, TestRunner
from chorus.testing.suites import SUITES

logger = logging.getLogger("chorus.commands.test")


def _result_embed(result: SuiteResult) -> discord.Embed:
    """Build a Discord embed from a SuiteResult."""
    if result.failed == 0:
        color = discord.Color.green()
    elif result.passed == 0:
        color = discord.Color.red()
    else:
        color = discord.Color.yellow()

    embed = discord.Embed(
        title=f"Suite: {result.suite_name}",
        description=(
            f"{result.passed} passed, {result.failed} failed | "
            f"{result.total_ms:.0f}ms total"
        ),
        color=color,
    )

    for tr in result.results:
        icon = "\u2705" if tr.passed else "\u274c"
        detail = tr.detail or tr.error or ""
        value = f"{icon} {tr.duration_ms:.0f}ms"
        if detail:
            value += f"\n{detail}"
        embed.add_field(name=tr.name, value=value, inline=False)

    return embed


class LiveTestCog(commands.Cog):
    """Cog for live testing commands. Only loaded when live_test_enabled is true."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        self._start_time = time.monotonic()

    test_group = app_commands.Group(name="test", description="Live testing commands")

    @test_group.command(name="run", description="Run a live test suite")
    @app_commands.describe(suite="Suite name (basic/throughput/agent/tools/all)")
    async def test_run(
        self, interaction: discord.Interaction, suite: str = "basic"
    ) -> None:
        if suite not in SUITES:
            available = ", ".join(sorted(SUITES.keys()))
            await interaction.response.send_message(
                f"Unknown suite: {suite!r}. Available: {available}", ephemeral=True
            )
            return

        test_channel = getattr(self.bot, "_test_channel", None)
        channel = test_channel or interaction.channel
        if channel is None:
            await interaction.response.send_message(
                "No test channel available.", ephemeral=True
            )
            return

        await interaction.response.defer()

        runner = TestRunner(self.bot, channel)  # type: ignore[arg-type]
        result = await runner.run_suite(suite, SUITES[suite])
        embed = _result_embed(result)
        await interaction.followup.send(embed=embed)

    @test_group.command(name="list", description="List available test suites")
    async def test_list(self, interaction: discord.Interaction) -> None:
        embed = discord.Embed(
            title="Available Test Suites",
            color=discord.Color.blurple(),
        )
        for name, tests in sorted(SUITES.items()):
            test_names = ", ".join(fn.__name__ for fn in tests)
            embed.add_field(
                name=f"{name} ({len(tests)} tests)",
                value=test_names,
                inline=False,
            )
        await interaction.response.send_message(embed=embed)

    @test_group.command(name="status", description="Show bot status")
    async def test_status(self, interaction: discord.Interaction) -> None:
        uptime_s = time.monotonic() - self._start_time
        hours, remainder = divmod(int(uptime_s), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours}h {minutes}m {seconds}s"

        latency_ms = self.bot.latency * 1000
        guild_count = len(self.bot.guilds)

        agent_count = 0
        if hasattr(self.bot, "agent_manager"):
            agents = await self.bot.agent_manager.list_agents()
            agent_count = len(agents)

        test_channel = getattr(self.bot, "_test_channel", None)
        channel_info = f"#{test_channel.name}" if test_channel else "Not configured"

        embed = discord.Embed(
            title="Bot Status",
            color=discord.Color.blurple(),
        )
        embed.add_field(name="Uptime", value=uptime_str, inline=True)
        embed.add_field(name="Latency", value=f"{latency_ms:.0f}ms", inline=True)
        embed.add_field(name="Guilds", value=str(guild_count), inline=True)
        embed.add_field(name="Agents", value=str(agent_count), inline=True)
        embed.add_field(name="Test Channel", value=channel_info, inline=True)

        await interaction.response.send_message(embed=embed)


async def setup(bot: commands.Bot) -> None:
    """Entry point for cog loading — only loads if live testing is enabled."""
    config = getattr(bot, "config", None)
    if config is None or not getattr(config, "live_test_enabled", False):
        logger.info("Live testing disabled — skipping LiveTestCog")
        return
    await bot.add_cog(LiveTestCog(bot))
    logger.info("LiveTestCog loaded")
