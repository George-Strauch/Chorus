"""Model shortcut slash commands — /haiku, /sonnet, /opus, /gpt-5, etc."""

from __future__ import annotations

import logging

import discord
from discord import app_commands
from discord.ext import commands

logger = logging.getLogger("chorus.commands.model")

MODEL_SHORTCUTS: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-5-20250929",
    "opus": "claude-opus-4-6",
    "gpt_4_1_nano": "gpt-4.1-nano",
    "gpt_5_nano": "gpt-5-nano",
    "gpt_5": "gpt-5",
    "gpt_5_pro": "gpt-5-pro",
    "gpt_5_2_pro": "gpt-5.2-pro",
    "o3_pro": "o3-pro",
    "o4_mini": "o4-mini",
}

# Display names for the prompt echo message
_DISPLAY_NAMES: dict[str, str] = {
    "haiku": "Haiku",
    "sonnet": "Sonnet",
    "opus": "Opus",
    "gpt_4_1_nano": "GPT-4.1 Nano",
    "gpt_5_nano": "GPT-5 Nano",
    "gpt_5": "GPT-5",
    "gpt_5_pro": "GPT-5 Pro",
    "gpt_5_2_pro": "GPT-5.2 Pro",
    "o3_pro": "o3-pro",
    "o4_mini": "o4-mini",
}


class ModelCog(commands.Cog):
    """Cog providing model shortcut slash commands."""

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    async def _run_model_shortcut(
        self,
        interaction: discord.Interaction,
        prompt: str,
        shortcut_key: str,
    ) -> None:
        """Shared handler for all model shortcut commands."""
        model_id = MODEL_SHORTCUTS[shortcut_key]
        display_name = _DISPLAY_NAMES[shortcut_key]

        await interaction.response.defer()

        # Post visible prompt message (slash command inputs are ephemeral)
        channel = interaction.channel
        assert channel is not None
        await channel.send(f"**[{display_name}]** {prompt}")  # type: ignore[union-attr]

        is_admin = False
        if isinstance(interaction.user, discord.Member):
            is_admin = interaction.user.guild_permissions.manage_guild

        try:
            await self.bot.run_model_query(  # type: ignore[attr-defined]
                channel=channel,
                author_id=interaction.user.id,
                is_admin=bool(is_admin),
                prompt=prompt,
                model_id=model_id,
            )
            await interaction.followup.send(
                f"Query sent to **{display_name}** (`{model_id}`).",
                ephemeral=True,
            )
        except ValueError as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
        except Exception:
            logger.exception("Model shortcut %s failed", shortcut_key)
            await interaction.followup.send(
                "An error occurred while processing your request.",
                ephemeral=True,
            )

    @app_commands.command(name="haiku", description="Query Claude Haiku 4.5")
    @app_commands.describe(prompt="Your prompt")
    async def haiku(self, interaction: discord.Interaction, prompt: str) -> None:
        await self._run_model_shortcut(interaction, prompt, "haiku")

    @app_commands.command(name="sonnet", description="Query Claude Sonnet 4.5")
    @app_commands.describe(prompt="Your prompt")
    async def sonnet(self, interaction: discord.Interaction, prompt: str) -> None:
        await self._run_model_shortcut(interaction, prompt, "sonnet")

    @app_commands.command(name="opus", description="Query Claude Opus 4.6")
    @app_commands.describe(prompt="Your prompt")
    async def opus(self, interaction: discord.Interaction, prompt: str) -> None:
        await self._run_model_shortcut(interaction, prompt, "opus")

    @app_commands.command(name="gpt-4-1-nano", description="Query GPT-4.1 Nano")
    @app_commands.describe(prompt="Your prompt")
    async def gpt_4_1_nano(self, interaction: discord.Interaction, prompt: str) -> None:
        await self._run_model_shortcut(interaction, prompt, "gpt_4_1_nano")

    @app_commands.command(name="gpt-5-nano", description="Query GPT-5 Nano")
    @app_commands.describe(prompt="Your prompt")
    async def gpt_5_nano(self, interaction: discord.Interaction, prompt: str) -> None:
        await self._run_model_shortcut(interaction, prompt, "gpt_5_nano")

    @app_commands.command(name="gpt-5", description="Query GPT-5")
    @app_commands.describe(prompt="Your prompt")
    async def gpt_5(self, interaction: discord.Interaction, prompt: str) -> None:
        await self._run_model_shortcut(interaction, prompt, "gpt_5")

    @app_commands.command(name="gpt-5-pro", description="Query GPT-5 Pro")
    @app_commands.describe(prompt="Your prompt")
    async def gpt_5_pro(self, interaction: discord.Interaction, prompt: str) -> None:
        await self._run_model_shortcut(interaction, prompt, "gpt_5_pro")

    @app_commands.command(name="gpt-5-2-pro", description="Query GPT-5.2 Pro")
    @app_commands.describe(prompt="Your prompt")
    async def gpt_5_2_pro(self, interaction: discord.Interaction, prompt: str) -> None:
        await self._run_model_shortcut(interaction, prompt, "gpt_5_2_pro")

    @app_commands.command(name="o3-pro", description="Query o3-pro")
    @app_commands.describe(prompt="Your prompt")
    async def o3_pro(self, interaction: discord.Interaction, prompt: str) -> None:
        await self._run_model_shortcut(interaction, prompt, "o3_pro")

    @app_commands.command(name="o4-mini", description="Query o4-mini")
    @app_commands.describe(prompt="Your prompt")
    async def o4_mini(self, interaction: discord.Interaction, prompt: str) -> None:
        await self._run_model_shortcut(interaction, prompt, "o4_mini")


async def setup(bot: commands.Bot) -> None:
    """Entry point for cog loading."""
    await bot.add_cog(ModelCog(bot))
