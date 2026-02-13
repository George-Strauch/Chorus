"""Settings slash commands — /settings group (model, permissions, show, validate-keys)."""

from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

from chorus.config import GlobalConfig, validate_model_available
from chorus.llm.discovery import read_cache, validate_and_discover
from chorus.permissions.engine import PRESETS

logger = logging.getLogger("chorus.commands.settings")


class SettingsCog(commands.Cog):
    """Cog for server-wide default settings management."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    settings_group = app_commands.Group(
        name="settings",
        description="Manage server-wide default settings",
        default_permissions=discord.Permissions(manage_guild=True),
    )

    @settings_group.command(name="model", description="Set the default LLM model")
    @app_commands.describe(model="Model name (e.g. claude-sonnet-4-20250514)")
    async def settings_model(self, interaction: discord.Interaction, model: str) -> None:
        chorus_home = self.bot.config.chorus_home  # type: ignore[attr-defined]
        if not validate_model_available(model, chorus_home):
            await interaction.response.send_message(
                f"Model **{model}** is not available. "
                "Run `/settings validate-keys` to refresh the model list.",
                ephemeral=True,
            )
            return

        gc: GlobalConfig = self.bot.global_config  # type: ignore[attr-defined]
        gc.default_model = model
        gc.save(chorus_home / "config.json")
        await interaction.response.send_message(
            f"Default model set to **{model}**.", ephemeral=True
        )

    @settings_group.command(name="permissions", description="Set the default permission profile")
    @app_commands.describe(profile="Permission preset name (open/standard/locked)")
    async def settings_permissions(
        self, interaction: discord.Interaction, profile: str
    ) -> None:
        if profile not in PRESETS:
            available = ", ".join(PRESETS)
            await interaction.response.send_message(
                f"Unknown profile **{profile}**. Available: {available}",
                ephemeral=True,
            )
            return

        gc: GlobalConfig = self.bot.global_config  # type: ignore[attr-defined]
        gc.default_permissions = profile
        gc.save(self.bot.config.chorus_home / "config.json")  # type: ignore[attr-defined]
        await interaction.response.send_message(
            f"Default permissions set to **{profile}**.", ephemeral=True
        )

    @settings_group.command(name="show", description="Display current settings")
    async def settings_show(self, interaction: discord.Interaction) -> None:
        gc: GlobalConfig = self.bot.global_config  # type: ignore[attr-defined]
        chorus_home = self.bot.config.chorus_home  # type: ignore[attr-defined]

        embed = discord.Embed(title="Chorus Settings", color=discord.Color.blurple())
        embed.add_field(
            name="Default Model",
            value=gc.default_model or "_Not set_",
            inline=False,
        )
        embed.add_field(name="Default Permissions", value=gc.default_permissions, inline=False)
        embed.add_field(
            name="Idle Timeout", value=f"{gc.idle_timeout}s ({gc.idle_timeout // 60}m)", inline=True
        )
        embed.add_field(
            name="Max Tool Iterations", value=str(gc.max_tool_loop_iterations), inline=True
        )
        embed.add_field(
            name="Max Bash Timeout", value=f"{gc.max_bash_timeout}s", inline=True
        )

        # API key status from cache
        cache = read_cache(chorus_home)
        if cache:
            providers = cache.get("providers", {})
            lines: list[str] = []
            for name, info in providers.items():
                valid = info.get("valid", False)
                models = info.get("models", [])
                status = f"Valid ({len(models)} models)" if valid else "Invalid"
                lines.append(f"**{name.capitalize()}:** {status}")
            if lines:
                embed.add_field(name="API Keys", value="\n".join(lines), inline=False)
            last_updated = cache.get("last_updated", "Unknown")
            embed.set_footer(text=f"Last key validation: {last_updated}")
        else:
            embed.add_field(
                name="API Keys",
                value="_No validation data. Run `/settings validate-keys`._",
                inline=False,
            )

        await interaction.response.send_message(embed=embed, ephemeral=True)

    @settings_group.command(
        name="validate-keys", description="Validate API keys and refresh available models"
    )
    async def settings_validate_keys(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)

        config = self.bot.config  # type: ignore[attr-defined]
        result = await validate_and_discover(
            config.chorus_home,
            anthropic_key=config.anthropic_api_key,
            openai_key=config.openai_api_key,
        )

        embed = discord.Embed(title="Key Validation Results", color=discord.Color.green())
        providers = result.get("providers", {})
        if not providers:
            embed.description = "No API keys configured."
        else:
            for name, info in providers.items():
                valid = info.get("valid", False)
                models = info.get("models", [])
                if valid:
                    model_list = ", ".join(models[:5])
                    if len(models) > 5:
                        model_list += f" (+{len(models) - 5} more)"
                    value = f"Valid — {len(models)} models\n{model_list}"
                else:
                    value = "Invalid key"
                embed.add_field(name=name.capitalize(), value=value, inline=False)

        embed.set_footer(text=f"Updated: {result.get('last_updated', 'now')}")
        await interaction.followup.send(embed=embed, ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    """Entry point for cog loading."""
    await bot.add_cog(SettingsCog(bot))
