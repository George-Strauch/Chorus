"""Process slash commands — /process list, /process kill, /process logs."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from chorus.process.models import (
    CallbackAction,
    ExitFilter,
    ProcessCallback,
    ProcessStatus,
    TrackedProcess,
    TriggerType,
)
from chorus.ui.process_embed import STATUS_COLORS

if TYPE_CHECKING:
    from chorus.process.manager import ProcessManager

logger = logging.getLogger("chorus.commands.process")

# Status emoji mapping
_STATUS_EMOJI = {
    ProcessStatus.RUNNING: "\U0001f7e2",  # 🟢
    ProcessStatus.EXITED: "\u26aa",  # ⚪
    ProcessStatus.KILLED: "\U0001f534",  # 🔴
    ProcessStatus.LOST: "\u26ab",  # ⚫
}

_MAX_EMBEDS = 10
_MAX_COMMAND_CHARS = 3800
_MAX_CONTEXT_CHARS = 1000


def _discord_timestamp(iso_string: str) -> str:
    """Convert an ISO 8601 timestamp to a Discord relative timestamp (<t:...:R>)."""
    try:
        dt = datetime.fromisoformat(iso_string)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return f"<t:{int(dt.timestamp())}:R>"
    except (ValueError, OSError):
        # Fallback to raw string if parsing fails
        return iso_string[:19]


def _safe_truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, adding '...' if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_hook(cb: ProcessCallback) -> str:
    """Format a single hook for display.

    Active hooks: **trigger_desc** → action\\n  fires: N/M
    Exhausted hooks: ~~trigger_desc → action~~ (exhausted)
    """
    # Build trigger description
    trigger = cb.trigger
    trigger_desc = f"**{trigger.type.value}**"
    if trigger.exit_filter != ExitFilter.ANY:
        trigger_desc += f"({trigger.exit_filter.value})"
    if trigger.pattern:
        trigger_desc += f" `{trigger.pattern}`"
    if trigger.timeout_seconds is not None:
        trigger_desc += f" {trigger.timeout_seconds}s"

    action_str = cb.action.value
    if cb.output_delay_seconds > 0:
        action_str += f" (delay {cb.output_delay_seconds}s)"

    if cb.exhausted:
        # Strip markdown bold for strikethrough
        plain_trigger = trigger_desc.replace("**", "")
        return f"~~{plain_trigger} \u2192 {action_str}~~ (exhausted)"

    line = f"{trigger_desc} \u2192 {action_str}"
    max_display = "\u221e" if cb.max_fires == 0 else str(cb.max_fires)
    line += f"\n\u2003fires: {cb.fire_count}/{max_display}"
    if cb.action == CallbackAction.NOTIFY_CHANNEL and cb.min_message_interval > 0:
        secs = cb.min_message_interval
        if secs >= 3600 and secs % 3600 == 0:
            interval_str = f"{int(secs // 3600)}h"
        elif secs >= 60 and secs % 60 == 0:
            interval_str = f"{int(secs // 60)}m"
        else:
            interval_str = f"{secs}s"
        line += f" interval: {interval_str}"
    return line


def _process_color(p: TrackedProcess) -> discord.Color:
    """Get the embed color for a process, with error-exit override."""
    color = STATUS_COLORS.get(p.status, discord.Color.greyple())
    if p.status == ProcessStatus.EXITED and p.exit_code is not None and p.exit_code != 0:
        color = discord.Color.red()
    return color


def _build_process_embed(p: TrackedProcess) -> discord.Embed:
    """Build a single embed for one process."""
    emoji = _STATUS_EMOJI.get(p.status, "\u2753")
    cmd_short = p.command
    if len(cmd_short) > 50:
        cmd_short = cmd_short[:47] + "..."
    title = f"{emoji} PID {p.pid} \u2014 {cmd_short}"

    # Full command in code block as description
    cmd_display = _safe_truncate(p.command, _MAX_COMMAND_CHARS)
    description = f"```\n{cmd_display}\n```"

    embed = discord.Embed(
        title=title,
        description=description,
        color=_process_color(p),
    )

    # Inline fields
    embed.add_field(name="Type", value=p.process_type.value, inline=True)
    embed.add_field(name="Started", value=_discord_timestamp(p.started_at), inline=True)

    if p.exit_code is not None:
        embed.add_field(name="Exit Code", value=str(p.exit_code), inline=True)

    # Working directory
    embed.add_field(name="Working Directory", value=f"`{p.working_directory}`", inline=False)

    # Hooks
    if p.callbacks:
        hook_lines = [_format_hook(cb) for cb in p.callbacks]
        hooks_text = "\n".join(hook_lines)
        embed.add_field(name="Hooks", value=_safe_truncate(hooks_text, 1024), inline=False)

    # Context
    if p.context:
        ctx_display = _safe_truncate(p.context, _MAX_CONTEXT_CHARS)
        embed.add_field(name="Context", value=ctx_display, inline=False)

    embed.set_footer(text=f"Agent: {p.agent_name}")
    return embed


def _build_overflow_embed(remaining: list[TrackedProcess]) -> discord.Embed:
    """Build a compact summary embed for processes that don't fit."""
    lines: list[str] = []
    for p in remaining:
        emoji = _STATUS_EMOJI.get(p.status, "\u2753")
        cmd_short = p.command
        if len(cmd_short) > 40:
            cmd_short = cmd_short[:37] + "..."
        exit_info = f" (exit {p.exit_code})" if p.exit_code is not None else ""
        lines.append(f"{emoji} **PID {p.pid}** \u2014 `{cmd_short}` [{p.status.value}{exit_info}]")

    return discord.Embed(
        title=f"+ {len(remaining)} more process(es)",
        description="\n".join(lines),
        color=discord.Color.light_grey(),
    )


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

        embeds: list[discord.Embed] = []

        if len(processes) <= _MAX_EMBEDS:
            # All fit as full embeds
            for p in processes:
                embeds.append(_build_process_embed(p))
        else:
            # First 9 as full embeds, 10th is overflow summary
            for p in processes[: _MAX_EMBEDS - 1]:
                embeds.append(_build_process_embed(p))
            embeds.append(_build_overflow_embed(processes[_MAX_EMBEDS - 1 :]))

        await interaction.response.send_message(embeds=embeds)

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
