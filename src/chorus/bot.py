"""Discord bot entrypoint for Chorus."""

from __future__ import annotations

import logging
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

import chorus.commands
from chorus.agent.context import ContextManager, build_llm_context
from chorus.agent.directory import AgentDirectory
from chorus.agent.manager import AgentManager
from chorus.agent.threads import ExecutionThread, ThreadManager, ThreadRunner, ThreadStatus
from chorus.config import BotConfig
from chorus.llm.providers import AnthropicProvider, OpenAIProvider
from chorus.llm.tool_loop import ToolExecutionContext, run_tool_loop
from chorus.permissions.engine import get_preset
from chorus.storage.db import Database
from chorus.tools.registry import create_default_registry

if TYPE_CHECKING:
    from chorus.agent.message_queue import ChannelMessageQueue
    from chorus.models import Agent

logger = logging.getLogger("chorus.bot")


class ChorusBot(commands.Bot):
    """Discord bot that turns channels into autonomous AI agents."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config

        # Thread and context management state (initialized early so on_message works pre-setup_hook)
        self._thread_managers: dict[str, ThreadManager] = {}
        self._context_managers: dict[str, ContextManager] = {}
        self._channel_to_agent: dict[int, str] = {}
        self._message_queues: dict[int, ChannelMessageQueue] = {}

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
        """Initialize storage, agent manager, load cogs, and register error handler."""
        # Initialize database
        self.db = Database(self.config.chorus_home / "db" / "chorus.db")
        await self.db.init()

        # Initialize agent manager
        template_dir = Path(__file__).resolve().parent.parent.parent / "template"
        directory = AgentDirectory(self.config.chorus_home, template_dir)
        directory.ensure_home()
        self.agent_manager = AgentManager(directory, self.db)

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

    async def close(self) -> None:
        """Kill all threads, shut down database, and disconnect."""
        for tm in self._thread_managers.values():
            await tm.kill_all()
        if hasattr(self, "db"):
            await self.db.close()
        await super().close()

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
        """Route messages to agent threads or process as commands."""
        if message.author == self.user:
            return

        # Check if this channel is bound to an agent
        agent_name = self._channel_to_agent.get(message.channel.id)
        if agent_name is None and hasattr(self, "agent_manager"):
            agent = await self.agent_manager.get_agent_by_channel(message.channel.id)
            if agent is not None:
                agent_name = agent.name
                self._channel_to_agent[message.channel.id] = agent_name

        if agent_name is None:
            await self.process_commands(message)
            return

        # Get or create ThreadManager for this agent
        tm = self._thread_managers.setdefault(
            agent_name, ThreadManager(agent_name, db=self.db)
        )

        # Get or create ContextManager for this agent
        if agent_name not in self._context_managers:
            agent_dir = self.agent_manager._directory._agents_dir / agent_name
            sessions_dir = agent_dir / "sessions"
            self._context_managers[agent_name] = ContextManager(
                agent_name, self.db, sessions_dir
            )
        cm = self._context_managers[agent_name]

        # Route: reply → existing thread, non-reply → new thread
        thread = None
        if message.reference and message.reference.message_id:
            thread = tm.route_message(message.reference.message_id)

        if thread is None:
            thread = tm.create_thread({"role": "user", "content": message.content})
        else:
            thread.messages.append({"role": "user", "content": message.content})

        # Persist user message to context
        await cm.persist_message(
            role="user",
            content=message.content,
            thread_id=thread.id,
            discord_message_id=message.id,
        )

        # Start if not running — wire the LLM tool loop as the runner
        if thread.status != ThreadStatus.RUNNING:
            agent = await self.agent_manager.get_agent_by_channel(message.channel.id)
            if agent is None:
                return

            runner = self._make_llm_runner(agent, tm, cm, message)
            tm.start_thread(thread, runner=runner)


    def _make_llm_runner(
        self,
        agent: Agent,
        tm: ThreadManager,
        cm: ContextManager,
        message: discord.Message,
    ) -> ThreadRunner:
        """Build a ThreadRunner closure that runs the LLM tool loop."""

        async def runner(thread: ExecutionThread) -> None:
            # Resolve permission profile
            profile = get_preset(agent.permissions)

            # Choose provider based on model name
            model = agent.model or "claude-sonnet-4-20250514"
            provider: AnthropicProvider | OpenAIProvider
            if model.startswith(("gpt-", "o1-", "o3-", "o4-")):
                if not self.config.openai_api_key:
                    await message.channel.send("No OpenAI API key configured.")
                    return
                provider = OpenAIProvider(self.config.openai_api_key, model)
            else:
                if not self.config.anthropic_api_key:
                    await message.channel.send("No Anthropic API key configured.")
                    return
                provider = AnthropicProvider(self.config.anthropic_api_key, model)

            # Build context
            agent_dir = self.agent_manager._directory._agents_dir / agent.name
            docs_dir = agent_dir / "docs"
            workspace = agent_dir / "workspace"

            ctx = ToolExecutionContext(
                workspace=workspace,
                profile=profile,
                agent_name=agent.name,
            )

            llm_messages = await build_llm_context(
                agent=agent,
                thread_id=thread.id,
                context_manager=cm,
                thread_manager=tm,
                agent_docs_dir=docs_dir,
            )

            registry = create_default_registry()

            # Run the tool loop
            result = await run_tool_loop(
                provider=provider,
                messages=llm_messages,
                tools=registry,
                ctx=ctx,
                system_prompt=agent.system_prompt,
                model=model,
            )

            # Send response to Discord
            if result.content:
                # Truncate to Discord's 2000 char limit
                content = result.content[:2000]
                bot_msg = await message.channel.send(content, reference=message)
                tm.register_bot_message(bot_msg.id, thread.id)

                # Persist assistant message
                await cm.persist_message(
                    role="assistant",
                    content=result.content,
                    thread_id=thread.id,
                    discord_message_id=bot_msg.id,
                )

        return runner


def main() -> None:
    """Entry point: load env, build config, run bot."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    config = BotConfig.from_env()
    bot = ChorusBot(config)
    bot.run(config.discord_token, log_handler=None)
