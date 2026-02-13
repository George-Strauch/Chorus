"""Branch slash commands — /branch list, /branch kill, /branch history."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

if TYPE_CHECKING:
    from chorus.agent.threads import ThreadManager

logger = logging.getLogger("chorus.commands.thread")


class ThreadCog(commands.Cog):
    """Cog for execution branch management commands."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    branch_group = app_commands.Group(name="branch", description="Execution branch management")

    def _get_thread_manager(self, channel_id: int) -> ThreadManager | None:
        """Look up the ThreadManager for a channel."""
        agent_name = getattr(self.bot, "_channel_to_agent", {}).get(channel_id)
        if agent_name is None:
            return None
        managers: dict[str, ThreadManager] = getattr(self.bot, "_thread_managers", {})
        return managers.get(agent_name)

    @branch_group.command(name="list", description="List active branches")
    async def branch_list(self, interaction: discord.Interaction) -> None:
        tm = self._get_thread_manager(interaction.channel.id)  # type: ignore[union-attr]
        if tm is None:
            await interaction.response.send_message(
                "No agent bound to this channel.", ephemeral=True
            )
            return

        threads = tm.list_all()
        if not threads:
            embed = discord.Embed(title="Branches", description="No branches.")
            await interaction.response.send_message(embed=embed)
            return

        embed = discord.Embed(title="Branches", description=f"{len(threads)} branch(es)")
        for t in threads:
            elapsed_s = t.metrics.elapsed_ms / 1000
            summary = t.summary or "Starting..."
            value = (
                f"Status: {t.status.value}\n"
                f"Step: {t.metrics.step_number} — {t.metrics.current_step}\n"
                f"Elapsed: {elapsed_s:.1f}s"
            )
            embed.add_field(name=f"#{t.id}: {summary}", value=value, inline=False)

        await interaction.response.send_message(embed=embed)

    @branch_group.command(name="kill", description="Kill a branch or all branches")
    @app_commands.describe(target="Branch ID or 'all'")
    async def branch_kill(self, interaction: discord.Interaction, target: str) -> None:
        tm = self._get_thread_manager(interaction.channel.id)  # type: ignore[union-attr]
        if tm is None:
            await interaction.response.send_message(
                "No agent bound to this channel.", ephemeral=True
            )
            return

        if target.lower() == "all":
            count = await tm.kill_all()
            await interaction.response.send_message(f"Killed {count} branch(es).")
            return

        try:
            branch_id = int(target)
        except ValueError:
            await interaction.response.send_message(
                f"Invalid target: {target!r}. Use a branch ID or 'all'.", ephemeral=True
            )
            return

        killed = await tm.kill_thread(branch_id)
        if killed:
            await interaction.response.send_message(f"Killed branch #{branch_id}.")
        else:
            await interaction.response.send_message(
                f"No branch #{branch_id} found.", ephemeral=True
            )

    @app_commands.command(
        name="break-context",
        description="Detach the main branch and start fresh",
    )
    async def break_context(self, interaction: discord.Interaction) -> None:
        tm = self._get_thread_manager(interaction.channel.id)  # type: ignore[union-attr]
        if tm is None:
            await interaction.response.send_message(
                "No agent bound to this channel.", ephemeral=True
            )
            return

        main = tm.get_main_thread()
        if main is None:
            await interaction.response.send_message("No active main branch to break.")
            return

        old_id = main.id
        tm.break_main_thread()
        await interaction.response.send_message(
            f"Detached main branch #{old_id}. Next message starts a fresh conversation."
        )

    @branch_group.command(name="history", description="Show step history for a branch")
    @app_commands.describe(branch_id="Branch ID")
    async def branch_history(self, interaction: discord.Interaction, branch_id: int) -> None:
        tm = self._get_thread_manager(interaction.channel.id)  # type: ignore[union-attr]
        if tm is None:
            await interaction.response.send_message(
                "No agent bound to this channel.", ephemeral=True
            )
            return

        thread = tm.get_thread(branch_id)
        if thread is None:
            await interaction.response.send_message(
                f"No branch #{branch_id} found.", ephemeral=True
            )
            return

        summary = thread.summary or "Unnamed"
        total_s = thread.metrics.elapsed_ms / 1000
        status = thread.status.value
        embed = discord.Embed(
            title=f"Branch #{branch_id} — {summary}",
            description=f"Status: {status}, {total_s:.1f}s total",
        )

        steps = thread.metrics.step_history
        if not steps:
            embed.add_field(name="Steps", value="No steps recorded.", inline=False)
        else:
            lines = []
            for step in steps:
                dur = f" ({step.duration_ms}ms)" if step.duration_ms is not None else " (running)"
                lines.append(f"Step {step.step_number}: {step.description}{dur}")
            embed.add_field(name="Steps", value="\n".join(lines), inline=False)

        await interaction.response.send_message(embed=embed)


async def setup(bot: commands.Bot) -> None:
    """Entry point for cog loading."""
    await bot.add_cog(ThreadCog(bot))
