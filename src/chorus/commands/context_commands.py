"""Context slash commands — /context clear, /context save, /context history, /context restore."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from chorus.models import SessionNotFoundError

if TYPE_CHECKING:
    from chorus.agent.context import ContextManager

logger = logging.getLogger("chorus.commands.context")


class ContextCog(commands.Cog):
    """Cog for context management commands."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    context_group = app_commands.Group(name="context", description="Context window management")

    def _get_context_manager(self, channel_id: int) -> ContextManager | None:
        """Look up the ContextManager for a channel."""
        agent_name = getattr(self.bot, "_channel_to_agent", {}).get(channel_id)
        if agent_name is None:
            return None
        managers: dict[str, ContextManager] = getattr(self.bot, "_context_managers", {})
        return managers.get(agent_name)

    @context_group.command(name="clear", description="Clear the context window")
    async def context_clear(self, interaction: discord.Interaction) -> None:
        cm = self._get_context_manager(interaction.channel.id)  # type: ignore[union-attr]
        if cm is None:
            await interaction.response.send_message(
                "No agent bound to this channel.", ephemeral=True
            )
            return

        await cm.clear()
        await interaction.response.send_message("Context cleared.")

    @context_group.command(name="save", description="Save a snapshot of the current context")
    @app_commands.describe(description="Optional description for the snapshot")
    async def context_save(
        self, interaction: discord.Interaction, description: str = ""
    ) -> None:
        cm = self._get_context_manager(interaction.channel.id)  # type: ignore[union-attr]
        if cm is None:
            await interaction.response.send_message(
                "No agent bound to this channel.", ephemeral=True
            )
            return

        meta = await cm.save_snapshot(description=description)
        await interaction.response.send_message(
            f"Context saved — session `{meta.session_id}` ({meta.message_count} messages)."
        )

    @context_group.command(name="history", description="List saved context sessions")
    async def context_history(self, interaction: discord.Interaction) -> None:
        cm = self._get_context_manager(interaction.channel.id)  # type: ignore[union-attr]
        if cm is None:
            await interaction.response.send_message(
                "No agent bound to this channel.", ephemeral=True
            )
            return

        snapshots = await cm.list_snapshots()
        if not snapshots:
            embed = discord.Embed(title="Context History", description="No saved sessions.")
            await interaction.response.send_message(embed=embed)
            return

        embed = discord.Embed(
            title="Context History", description=f"{len(snapshots)} session(s)"
        )
        for snap in snapshots:
            summary = snap.summary or "(no summary)"
            value = (
                f"Description: {snap.description or '(none)'}\n"
                f"Summary: {summary}\n"
                f"Messages: {snap.message_count}\n"
                f"Saved: {snap.saved_at}"
            )
            embed.add_field(
                name=f"`{snap.session_id[:8]}...`",
                value=value,
                inline=False,
            )

        await interaction.response.send_message(embed=embed)

    @context_group.command(name="restore", description="Restore a saved context session")
    @app_commands.describe(session_id="Session ID to restore")
    async def context_restore(
        self, interaction: discord.Interaction, session_id: str
    ) -> None:
        cm = self._get_context_manager(interaction.channel.id)  # type: ignore[union-attr]
        if cm is None:
            await interaction.response.send_message(
                "No agent bound to this channel.", ephemeral=True
            )
            return

        try:
            await cm.restore_snapshot(session_id)
        except SessionNotFoundError:
            await interaction.response.send_message(
                f"Session `{session_id}` not found.", ephemeral=True
            )
            return

        await interaction.response.send_message(
            f"Session `{session_id}` restored."
        )


async def setup(bot: commands.Bot) -> None:
    """Entry point for cog loading."""
    await bot.add_cog(ContextCog(bot))
