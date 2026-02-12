"""Discord bot entrypoint for Chorus."""

from __future__ import annotations

import logging
import pkgutil

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

import chorus.commands
from chorus.config import BotConfig

logger = logging.getLogger("chorus.bot")


class ChorusBot(commands.Bot):
    """Discord bot that turns channels into autonomous AI agents."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config

        intents = discord.Intents.default()
        intents.guilds = True
        intents.guild_messages = True
        intents.message_content = True
        intents.presences = False
        intents.members = False

        super().__init__(
            command_prefix="!",
            intents=intents,
        )

    async def setup_hook(self) -> None:
        """Load cogs and register error handler."""
        # Auto-discover and load all command cogs
        for module_info in pkgutil.iter_modules(chorus.commands.__path__):
            ext = f"chorus.commands.{module_info.name}"
            try:
                await self.load_extension(ext)
                logger.info("Loaded extension: %s", ext)
            except Exception:
                logger.exception("Failed to load extension: %s", ext)

        # Register tree error handler
        @self.tree.error
        async def on_app_command_error(
            interaction: discord.Interaction,
            error: app_commands.AppCommandError,
        ) -> None:
            if isinstance(error, app_commands.MissingPermissions):
                msg = "You don't have permission to use this command."
            elif isinstance(error, app_commands.CommandNotFound):
                msg = "Command not found."
            elif isinstance(error, app_commands.CommandInvokeError):
                logger.exception("Command error", exc_info=error.original)
                msg = "An internal error occurred."
            else:
                logger.exception("Unhandled app command error", exc_info=error)
                msg = "An unexpected error occurred."
            if not interaction.response.is_done():
                await interaction.response.send_message(msg, ephemeral=True)
            else:
                await interaction.followup.send(msg, ephemeral=True)

    async def on_ready(self) -> None:
        """Log startup info and sync command tree."""
        assert self.user is not None
        logger.info(
            "Logged in as %s (id=%s) | guilds=%d | cogs=%d",
            self.user.name,
            self.user.id,
            len(self.guilds),
            len(self.cogs),
        )
        if self.config.dev_guild_id:
            guild = discord.Object(id=self.config.dev_guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info("Commands synced to dev guild %s", self.config.dev_guild_id)
        else:
            await self.tree.sync()
            logger.info("Commands synced globally")

    async def on_message(self, message: discord.Message) -> None:
        """Ignore own messages, process commands for others."""
        if message.author == self.user:
            return
        await self.process_commands(message)


def main() -> None:
    """Entry point: load env, build config, run bot."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    config = BotConfig.from_env()
    bot = ChorusBot(config)
    bot.run(config.discord_token, log_handler=None)
