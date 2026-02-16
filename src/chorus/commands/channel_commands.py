"""Channel management slash commands — /purge."""

from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

logger = logging.getLogger("chorus.commands.channel")

# discord.TextChannel.purge() handles the 14-day / bulk-delete limit
# internally — messages older than 14 days are deleted one-by-one.
_PURGE_LIMIT = 5000


class ChannelCog(commands.Cog):
    """Cog for channel management commands."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(
        name="purge",
        description="Delete all messages in this agent channel",
    )
    async def purge(self, interaction: discord.Interaction) -> None:
        # Must be in an agent channel
        channel_to_agent: dict[int, str] = getattr(self.bot, "_channel_to_agent", {})
        channel_id = interaction.channel.id  # type: ignore[union-attr]

        agent_name = channel_to_agent.get(channel_id)
        if agent_name is None:
            # Try DB lookup (channel may not be cached yet)
            if hasattr(self.bot, "agent_manager"):
                agent = await self.bot.agent_manager.get_agent_by_channel(channel_id)
                if agent is not None:
                    agent_name = agent.name

        if agent_name is None:
            await interaction.response.send_message(
                "This command can only be used in an agent channel.", ephemeral=True
            )
            return

        channel = interaction.channel
        if not isinstance(channel, discord.TextChannel):
            await interaction.response.send_message(
                "This command only works in text channels.", ephemeral=True
            )
            return

        # Check bot has manage_messages permission
        bot_member = channel.guild.me
        if bot_member is None or not channel.permissions_for(bot_member).manage_messages:
            await interaction.response.send_message(
                "I need **Manage Messages** permission to purge this channel.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=True)

        try:
            deleted = await channel.purge(limit=_PURGE_LIMIT)
            count = len(deleted)
            await interaction.followup.send(
                f"Purged {count} message{'s' if count != 1 else ''} from #{channel.name}.",
                ephemeral=True,
            )
            logger.info(
                "Purged %d messages from #%s (agent: %s, user: %s)",
                count, channel.name, agent_name, interaction.user,
            )
        except discord.Forbidden:
            await interaction.followup.send(
                "Missing permissions to delete messages.", ephemeral=True
            )
        except discord.HTTPException as exc:
            logger.warning("Purge failed for #%s: %s", channel.name, exc)
            await interaction.followup.send(
                f"Purge failed: {exc}", ephemeral=True
            )


async def setup(bot: commands.Bot) -> None:
    """Entry point for cog loading."""
    await bot.add_cog(ChannelCog(bot))
