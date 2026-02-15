"""ProcessStatusEmbed — live-updating Discord embed for background processes."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from typing import Any

import discord

from chorus.process.models import ProcessStatus

logger = logging.getLogger("chorus.ui.process_embed")

# Color mapping
_COLORS = {
    ProcessStatus.RUNNING: discord.Color.green(),
    ProcessStatus.EXITED: discord.Color.greyple(),
    ProcessStatus.KILLED: discord.Color.red(),
    ProcessStatus.LOST: discord.Color.dark_grey(),
}

_UPDATE_INTERVAL_S = 5.0
_MAX_OUTPUT_LINES = 10
_MAX_OUTPUT_CHARS = 900


class ProcessStatusEmbed:
    """Manages a live-updating Discord embed for a tracked process.

    The embed shows the process command, PID, uptime, status,
    and last few lines of output. Updates every 5 seconds while
    the process is running.

    Parameters
    ----------
    channel:
        Discord channel to send/update the embed in.
    pid:
        Process ID to track.
    command:
        The command string (for display).
    agent_name:
        Agent that owns the process.
    process_manager:
        Reference to get live process data.
    """

    def __init__(
        self,
        channel: Any,
        pid: int,
        command: str,
        agent_name: str,
        process_manager: Any,
    ) -> None:
        self._channel = channel
        self._pid = pid
        self._command = command
        self._agent_name = agent_name
        self._pm = process_manager
        self._message: discord.Message | None = None
        self._ticker_task: asyncio.Task[None] | None = None
        self._started_at = datetime.now(UTC)

    @property
    def message(self) -> discord.Message | None:
        return self._message

    async def start(self) -> discord.Message | None:
        """Send the initial embed and start the update ticker."""
        embed = self._build_embed()
        try:
            self._message = await self._channel.send(embed=embed)
        except Exception:
            logger.warning("Failed to send process embed", exc_info=True)
            return None

        self._ticker_task = asyncio.create_task(self._ticker_loop())
        return self._message

    async def stop(self) -> None:
        """Stop the update ticker and do a final update."""
        if self._ticker_task is not None and not self._ticker_task.done():
            self._ticker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ticker_task

        # Final update
        await self._update_embed()

    async def _ticker_loop(self) -> None:
        """Periodically update the embed."""
        try:
            while True:
                await asyncio.sleep(_UPDATE_INTERVAL_S)
                tracked = self._pm.get_process(self._pid)
                if tracked is None:
                    break

                await self._update_embed()

                if tracked.status != ProcessStatus.RUNNING:
                    break
        except asyncio.CancelledError:
            pass

    async def _update_embed(self) -> None:
        """Edit the message with current process state."""
        if self._message is None:
            return
        embed = self._build_embed()
        try:
            await self._message.edit(embed=embed)
        except Exception:
            logger.debug("Failed to update process embed", exc_info=True)

    def _build_embed(self) -> discord.Embed:
        """Build the process status embed."""
        tracked = self._pm.get_process(self._pid)

        if tracked is None:
            return discord.Embed(
                title=f"Process {self._pid}",
                description="Process not found.",
                color=discord.Color.dark_grey(),
            )

        status = tracked.status
        color = _COLORS.get(status, discord.Color.greyple())

        # Override color for error exits
        if status == ProcessStatus.EXITED and tracked.exit_code != 0:
            color = discord.Color.red()

        # Uptime
        elapsed = (datetime.now(UTC) - self._started_at).total_seconds()
        if elapsed < 60:
            uptime = f"{elapsed:.0f}s"
        elif elapsed < 3600:
            uptime = f"{elapsed / 60:.1f}m"
        else:
            uptime = f"{elapsed / 3600:.1f}h"

        # Command preview
        cmd = self._command
        if len(cmd) > 60:
            cmd = cmd[:57] + "..."

        embed = discord.Embed(
            title=f"Process {self._pid}",
            color=color,
        )
        embed.add_field(name="Command", value=f"`{cmd}`", inline=False)
        embed.add_field(name="Status", value=status.value, inline=True)
        embed.add_field(name="Uptime", value=uptime, inline=True)
        embed.add_field(name="Type", value=tracked.process_type.value, inline=True)

        if tracked.exit_code is not None:
            embed.add_field(
                name="Exit Code",
                value=str(tracked.exit_code),
                inline=True,
            )

        # Rolling tail output
        if tracked.rolling_tail:
            tail = list(tracked.rolling_tail)[-_MAX_OUTPUT_LINES:]
            output = "\n".join(tail)
            if len(output) > _MAX_OUTPUT_CHARS:
                output = "..." + output[-_MAX_OUTPUT_CHARS:]
            embed.add_field(
                name="Recent Output",
                value=f"```\n{output}\n```",
                inline=False,
            )

        # Active watchers
        active_watchers = [
            cb for cb in tracked.callbacks if not cb.exhausted
        ]
        if active_watchers:
            watcher_lines = []
            for cb in active_watchers[:5]:
                trigger = cb.trigger
                if trigger.pattern:
                    watcher_lines.append(
                        f"/{trigger.pattern}/ → {cb.action.value}"
                    )
                else:
                    watcher_lines.append(
                        f"{trigger.type.value} → {cb.action.value}"
                    )
            embed.add_field(
                name="Active Watchers",
                value="\n".join(watcher_lines),
                inline=False,
            )

        embed.set_footer(text=f"Agent: {self._agent_name}")
        return embed
