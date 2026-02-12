"""Agent slash commands — /ping and /agent group (init, destroy, list, config)."""

from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

from chorus.models import AgentExistsError, AgentNotFoundError, InvalidAgentNameError

logger = logging.getLogger("chorus.commands.agent")


class ConfirmDestroyView(discord.ui.View):
    """Confirmation view for agent destruction."""

    def __init__(self, bot: commands.Bot, name: str) -> None:
        super().__init__(timeout=30)
        self._bot = bot
        self._name = name

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.danger)
    async def confirm(
        self, interaction: discord.Interaction, button: discord.ui.Button[ConfirmDestroyView]
    ) -> None:
        try:
            await self._bot.agent_manager.destroy(self._name)  # type: ignore[attr-defined]
            await interaction.response.send_message(
                f"Agent **{self._name}** destroyed.", ephemeral=True
            )
        except AgentNotFoundError:
            await interaction.response.send_message(
                f"Agent **{self._name}** not found.", ephemeral=True
            )
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(
        self, interaction: discord.Interaction, button: discord.ui.Button[ConfirmDestroyView]
    ) -> None:
        await interaction.response.send_message("Cancelled.", ephemeral=True)
        self.stop()


class AgentCog(commands.Cog):
    """Cog for agent management commands."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="ping", description="Check bot latency")
    async def ping(self, interaction: discord.Interaction) -> None:
        latency_ms = round(self.bot.latency * 1000, 1)
        await interaction.response.send_message(f"Pong! {latency_ms}ms")

    agent_group = app_commands.Group(name="agent", description="Agent management")

    @agent_group.command(name="init", description="Create a new agent")
    @app_commands.describe(
        name="Agent name (lowercase, hyphens, 2-32 chars)",
        system_prompt="Custom system prompt",
        model="LLM model to use",
        permissions="Permission profile (open/standard/locked)",
    )
    async def agent_init(
        self,
        interaction: discord.Interaction,
        name: str,
        system_prompt: str | None = None,
        model: str | None = None,
        permissions: str | None = None,
    ) -> None:
        # Find or create the "Chorus Agents" category
        category = discord.utils.get(interaction.guild.categories, name="Chorus Agents")  # type: ignore[union-attr]
        if not category:
            category = await interaction.guild.create_category("Chorus Agents")  # type: ignore[union-attr]

        channel = await interaction.guild.create_text_channel(name, category=category)  # type: ignore[union-attr]

        overrides: dict[str, str] = {}
        if system_prompt:
            overrides["system_prompt"] = system_prompt
        if model:
            overrides["model"] = model
        if permissions:
            overrides["permissions"] = permissions

        try:
            agent = await self.bot.agent_manager.create(  # type: ignore[attr-defined]
                name,
                guild_id=interaction.guild_id,
                channel_id=channel.id,
                overrides=overrides or None,
            )
            await interaction.response.send_message(
                f"Agent **{agent.name}** created in {channel.mention}"
            )
        except (InvalidAgentNameError, AgentExistsError) as exc:
            # Clean up the channel we just created
            await channel.delete()
            await interaction.response.send_message(str(exc), ephemeral=True)

    @agent_group.command(name="destroy", description="Destroy an agent")
    @app_commands.describe(name="Agent name to destroy")
    async def agent_destroy(self, interaction: discord.Interaction, name: str) -> None:
        view = ConfirmDestroyView(self.bot, name)
        await interaction.response.send_message(
            f"Destroy agent **{name}**? This cannot be undone.",
            view=view,
            ephemeral=True,
        )

    @agent_group.command(name="list", description="List all agents")
    async def agent_list(self, interaction: discord.Interaction) -> None:
        agents = await self.bot.agent_manager.list_agents(interaction.guild_id)  # type: ignore[attr-defined]
        if not agents:
            embed = discord.Embed(title="Agents", description="No agents created yet.")
        else:
            embed = discord.Embed(title="Agents", description=f"{len(agents)} agent(s)")
            for a in agents:
                embed.add_field(
                    name=a.name,
                    value=f"<#{a.channel_id}> | {a.permissions}",
                    inline=False,
                )
        await interaction.response.send_message(embed=embed)

    @agent_group.command(name="config", description="Configure an agent")
    @app_commands.describe(
        name="Agent name",
        key="Config key (system_prompt, model, permissions)",
        value="New value",
    )
    async def agent_config(
        self, interaction: discord.Interaction, name: str, key: str, value: str
    ) -> None:
        try:
            await self.bot.agent_manager.configure(name, key, value)  # type: ignore[attr-defined]
            await interaction.response.send_message(
                f"Updated **{name}**.{key}", ephemeral=True
            )
        except (ValueError, AgentNotFoundError) as exc:
            await interaction.response.send_message(str(exc), ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    """Entry point for cog loading."""
    await bot.add_cog(AgentCog(bot))
