"""Process slash commands — /process list, /process kill, /process logs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from chorus.process.models import ExitFilter

if TYPE_CHECKING:
    from chorus.process.manager import ProcessManager

logger = logging.getLogger("chorus.commands.process")


class ProcessCog(commands.Cog):
    """Cog for process management commands."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    process_group = app_commands.Group(name="process", description="Process management")

    def _get_agent_name(self, channel_id: int) -> str | None:
        """Look up the agent bound to a channel."""
        return getattr(self.bot, "_channel_to_agent", {}).get(channel_id)

    def _get_process_manager(self) -> ProcessManager | None:
        """Get the ProcessManager from the bot."""
        return getattr(self.bot, "_process_manager", None)

    @process_group.command(name="list", description="List running/recent processes")
    async def process_list(self, interaction: discord.Interaction) -> None:
        pm = self._get_process_manager()
        if pm is None:
            await interaction.response.send_message(
                "Process manager not available.", ephemeral=True
            )
            return

        agent_name = self._get_agent_name(interaction.channel.id)  # type: ignore[union-attr]
        processes = pm.list_processes(agent_name)

        if not processes:
            await interaction.response.send_message(
                "No processes tracked for this agent.", ephemeral=True
            )
            return

        embed = discord.Embed(
            title="Processes",
            description=f"{len(processes)} process(es)",
        )
        for p in processes[:20]:  # Cap at 20
            # Full command in code block (capped at embed field limit)
            cmd_display = p.command
            if len(cmd_display) > 900:
                cmd_display = cmd_display[:900] + "..."
            status = p.status.value
            exit_info = f" (exit {p.exit_code})" if p.exit_code is not None else ""

            lines = [
                f"```\n{cmd_display}\n```",
                f"Type: {p.process_type.value} | Status: {status}{exit_info}",
                f"Started: {p.started_at[:19]}",
            ]

            # Show callbacks
            active_cbs = [cb for cb in p.callbacks if not cb.exhausted]
            if active_cbs:
                cb_lines: list[str] = []
                for cb in active_cbs:
                    trigger_desc = cb.trigger.type.value
                    if cb.trigger.exit_filter != ExitFilter.ANY:
                        trigger_desc += f"({cb.trigger.exit_filter.value})"
                    if cb.trigger.pattern:
                        trigger_desc += f" `{cb.trigger.pattern}`"
                    if cb.trigger.timeout_seconds is not None:
                        trigger_desc += f" {cb.trigger.timeout_seconds}s"
                    cb_lines.append(f"{trigger_desc} \u2192 {cb.action.value}")
                lines.append("Hooks: " + "; ".join(cb_lines))

            # Show context preview
            if p.context:
                ctx_preview = p.context[:100]
                if len(p.context) > 100:
                    ctx_preview += "..."
                lines.append(f"Context: {ctx_preview}")

            value = "\n".join(lines)
            # Discord embed field value max is 1024 chars
            if len(value) > 1024:
                value = value[:1021] + "..."
            embed.add_field(name=f"PID {p.pid}", value=value, inline=False)

        await interaction.response.send_message(embed=embed)

    @process_group.command(name="kill", description="Kill a running process")
    @app_commands.describe(pid="Process ID to kill")
    async def process_kill(self, interaction: discord.Interaction, pid: int) -> None:
        pm = self._get_process_manager()
        if pm is None:
            await interaction.response.send_message(
                "Process manager not available.", ephemeral=True
            )
            return

        # Verify process belongs to this channel's agent
        agent_name = self._get_agent_name(interaction.channel.id)  # type: ignore[union-attr]
        process = pm.get_process(pid)
        if process is None:
            await interaction.response.send_message(
                f"No process with PID {pid} found.", ephemeral=True
            )
            return
        if agent_name is not None and process.agent_name != agent_name:
            await interaction.response.send_message(
                f"Process {pid} belongs to a different agent.", ephemeral=True
            )
            return

        killed = await pm.kill_process(pid)
        if killed:
            await interaction.response.send_message(f"Killed process {pid}.")
        else:
            await interaction.response.send_message(
                f"Process {pid} is not running.", ephemeral=True
            )

    @process_group.command(name="logs", description="Show recent output from a process")
    @app_commands.describe(pid="Process ID", lines="Number of lines to show (default 20)")
    async def process_logs(
        self, interaction: discord.Interaction, pid: int, lines: int = 20
    ) -> None:
        pm = self._get_process_manager()
        if pm is None:
            await interaction.response.send_message(
                "Process manager not available.", ephemeral=True
            )
            return

        process = pm.get_process(pid)
        if process is None:
            await interaction.response.send_message(
                f"No process with PID {pid} found.", ephemeral=True
            )
            return

        # Get rolling tail
        tail = list(process.rolling_tail)
        if not tail:
            await interaction.response.send_message(
                f"No output captured for PID {pid}.", ephemeral=True
            )
            return

        # Take last N lines
        display = tail[-lines:]
        output = "\n".join(display)

        # Truncate if too long for Discord
        if len(output) > 1900:
            output = output[-1900:]
            output = "...(truncated)\n" + output

        await interaction.response.send_message(
            f"**Process {pid} output** ({len(display)} lines):\n```\n{output}\n```"
        )


async def setup(bot: commands.Bot) -> None:
    """Entry point for cog loading."""
    await bot.add_cog(ProcessCog(bot))
