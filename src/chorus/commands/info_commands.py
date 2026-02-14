"""Info slash commands — /models and /permissions (top-level, no permissions required)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from chorus.llm.discovery import read_cache
from chorus.permissions.engine import PRESETS

if TYPE_CHECKING:
    from chorus.config import GlobalConfig

logger = logging.getLogger("chorus.commands.info")


class InfoCog(commands.Cog):
    """Cog for informational commands available to all users."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="models", description="List available LLM models")
    async def models_command(self, interaction: discord.Interaction) -> None:
        chorus_home = self.bot.config.chorus_home  # type: ignore[attr-defined]
        cache = read_cache(chorus_home)

        if cache is None:
            await interaction.response.send_message(
                "No model data yet. Run `/settings validate-keys` to discover models.",
                ephemeral=True,
            )
            return

        gc: GlobalConfig = self.bot.global_config  # type: ignore[attr-defined]
        current_default = gc.default_model

        embed = discord.Embed(
            title="Available Models", color=discord.Color.blurple()
        )

        providers = cache.get("providers", {})
        total = 0
        for provider_name, info in providers.items():
            if not info.get("valid"):
                embed.add_field(
                    name=provider_name.capitalize(),
                    value="_Invalid API key_",
                    inline=False,
                )
                continue

            models = info.get("models", [])
            total += len(models)
            if not models:
                embed.add_field(
                    name=provider_name.capitalize(),
                    value="_No models found_",
                    inline=False,
                )
                continue

            lines: list[str] = []
            for m in models:
                prefix = "**>** " if m == current_default else "- "
                lines.append(f"{prefix}`{m}`")

            # Discord embed field value limit is 1024 chars
            text = "\n".join(lines)
            if len(text) > 1024:
                truncated: list[str] = []
                length = 0
                for line in lines:
                    if length + len(line) + 1 > 980:
                        break
                    truncated.append(line)
                    length += len(line) + 1
                remaining = len(models) - len(truncated)
                text = "\n".join(truncated)
                text += f"\n_...and {remaining} more_"

            embed.add_field(
                name=f"{provider_name.capitalize()} ({len(models)})",
                value=text,
                inline=False,
            )

        if not providers:
            embed.description = (
                "No providers configured. "
                "Run `/settings validate-keys` to discover models."
            )

        if current_default:
            embed.set_footer(text=f"Current default: {current_default}")
        else:
            embed.set_footer(text="No default model set")

        await interaction.response.send_message(embed=embed, ephemeral=True)

    @app_commands.command(
        name="permissions", description="List available permission presets"
    )
    async def permissions_command(
        self, interaction: discord.Interaction
    ) -> None:
        gc: GlobalConfig = self.bot.global_config  # type: ignore[attr-defined]
        current_default = gc.default_permissions

        embed = discord.Embed(
            title="Permission Presets", color=discord.Color.blurple()
        )

        for name, profile in PRESETS.items():
            marker = " (default)" if name == current_default else ""
            allow = ", ".join(f"`{p}`" for p in profile.allow) or "_none_"
            ask = ", ".join(f"`{p}`" for p in profile.ask) or "_none_"
            embed.add_field(
                name=f"{name}{marker}",
                value=f"**Allow:** {allow}\n**Ask:** {ask}",
                inline=False,
            )

        embed.set_footer(text=f"Current default: {current_default}")
        await interaction.response.send_message(embed=embed, ephemeral=True)


async def setup(bot: commands.Bot) -> None:
    """Entry point for cog loading."""
    await bot.add_cog(InfoCog(bot))
