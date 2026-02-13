"""Thread slash commands — /thread list, /thread kill, /thread history."""

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
    """Cog for execution thread management commands."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    thread_group = app_commands.Group(name="thread", description="Execution thread management")

    def _get_thread_manager(self, channel_id: int) -> ThreadManager | None:
        """Look up the ThreadManager for a channel."""
        agent_name = getattr(self.bot, "_channel_to_agent", {}).get(channel_id)
        if agent_name is None:
            return None
        managers: dict[str, ThreadManager] = getattr(self.bot, "_thread_managers", {})
        return managers.get(agent_name)

    @thread_group.command(name="list", description="List active threads")
    async def thread_list(self, interaction: discord.Interaction) -> None:
        tm = self._get_thread_manager(interaction.channel.id)  # type: ignore[union-attr]
        if tm is None:
            await interaction.response.send_message(
                "No agent bound to this channel.", ephemeral=True
            )
            return

        threads = tm.list_all()
        if not threads:
            embed = discord.Embed(title="Threads", description="No threads.")
            await interaction.response.send_message(embed=embed)
            return

        embed = discord.Embed(title="Threads", description=f"{len(threads)} thread(s)")
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

    @thread_group.command(name="kill", description="Kill a thread or all threads")
    @app_commands.describe(target="Thread ID or 'all'")
    async def thread_kill(self, interaction: discord.Interaction, target: str) -> None:
        tm = self._get_thread_manager(interaction.channel.id)  # type: ignore[union-attr]
        if tm is None:
            await interaction.response.send_message(
                "No agent bound to this channel.", ephemeral=True
            )
            return

        if target.lower() == "all":
            count = await tm.kill_all()
            await interaction.response.send_message(f"Killed {count} thread(s).")
            return

        try:
            thread_id = int(target)
        except ValueError:
            await interaction.response.send_message(
                f"Invalid target: {target!r}. Use a thread ID or 'all'.", ephemeral=True
            )
            return

        killed = await tm.kill_thread(thread_id)
        if killed:
            await interaction.response.send_message(f"Killed thread #{thread_id}.")
        else:
            await interaction.response.send_message(
                f"No thread #{thread_id} found.", ephemeral=True
            )

    @app_commands.command(
        name="break-context",
        description="Detach the main thread and start fresh",
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
            await interaction.response.send_message("No active main thread to break.")
            return

        old_id = main.id
        tm.break_main_thread()
        await interaction.response.send_message(
            f"Detached main thread #{old_id}. Next message starts a fresh conversation."
        )

    @thread_group.command(name="history", description="Show step history for a thread")
    @app_commands.describe(thread_id="Thread ID")
    async def thread_history(self, interaction: discord.Interaction, thread_id: int) -> None:
        tm = self._get_thread_manager(interaction.channel.id)  # type: ignore[union-attr]
        if tm is None:
            await interaction.response.send_message(
                "No agent bound to this channel.", ephemeral=True
            )
            return

        thread = tm.get_thread(thread_id)
        if thread is None:
            await interaction.response.send_message(
                f"No thread #{thread_id} found.", ephemeral=True
            )
            return

        summary = thread.summary or "Unnamed"
        total_s = thread.metrics.elapsed_ms / 1000
        status = thread.status.value
        embed = discord.Embed(
            title=f"Thread #{thread_id} — {summary}",
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
