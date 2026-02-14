"""Agent slash commands — /ping and /agent group (init, destroy, list, config)."""

from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

from chorus.agent.manager import CONFIGURABLE_KEYS
from chorus.llm.discovery import get_cached_models
from chorus.models import AgentExistsError, AgentNotFoundError, InvalidAgentNameError
from chorus.permissions.engine import PRESETS

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
            # Look up agent to get channel_id before destroying
            agent_data = await self._bot.agent_manager._db.get_agent(self._name)  # type: ignore[attr-defined]
            if agent_data and interaction.guild:
                channel = interaction.guild.get_channel(int(agent_data["channel_id"]))
                if channel is not None:
                    await channel.delete(reason=f"Agent {self._name} destroyed")

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
        # Defer — agent creation includes an LLM call for prompt refinement
        await interaction.response.defer()

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
            await interaction.followup.send(
                f"Agent **{agent.name}** created in {channel.mention}"
            )

            # Send welcome message to the new channel
            model_display = agent.model or "server default"
            welcome = (
                f"**{agent.name}** is ready.\n\n"
                f"**Model:** `{model_display}`\n"
                f"**Permissions:** `{agent.permissions}`\n\n"
                "Just type a message to start talking. "
                "This channel is the agent's workspace — "
                "it can read/write files, run commands, and manage its own config.\n\n"
                "**Useful commands:**\n"
                "- `/agent config` — change model, prompt, or permissions\n"
                "- `/context clear` — reset the conversation window\n"
                "- `/thread list` — see running execution threads\n"
                "- `/status` — agent overview\n"
            )
            await channel.send(welcome)
        except (InvalidAgentNameError, AgentExistsError) as exc:
            # Clean up the channel we just created
            await channel.delete()
            await interaction.followup.send(str(exc), ephemeral=True)

    @agent_init.autocomplete("model")
    async def _init_model_autocomplete(
        self, interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        chorus_home = self.bot.config.chorus_home  # type: ignore[attr-defined]
        models = get_cached_models(chorus_home)
        filtered = [m for m in models if current.lower() in m.lower()]
        return [app_commands.Choice(name=m, value=m) for m in filtered[:25]]

    @agent_init.autocomplete("permissions")
    async def _init_permissions_autocomplete(
        self, interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        return [
            app_commands.Choice(name=p, value=p)
            for p in PRESETS
            if current.lower() in p.lower()
        ]

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


    @agent_config.autocomplete("key")
    async def _config_key_autocomplete(
        self, interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        return [
            app_commands.Choice(name=k, value=k)
            for k in sorted(CONFIGURABLE_KEYS)
            if current.lower() in k.lower()
        ]

    @agent_config.autocomplete("value")
    async def _config_value_autocomplete(
        self, interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        key = interaction.namespace.key
        if key == "model":
            chorus_home = self.bot.config.chorus_home  # type: ignore[attr-defined]
            models = get_cached_models(chorus_home)
            filtered = [m for m in models if current.lower() in m.lower()]
            return [app_commands.Choice(name=m, value=m) for m in filtered[:25]]
        elif key == "permissions":
            return [
                app_commands.Choice(name=p, value=p)
                for p in PRESETS
                if current.lower() in p.lower()
            ]
        return []


async def setup(bot: commands.Bot) -> None:
    """Entry point for cog loading."""
    await bot.add_cog(AgentCog(bot))
