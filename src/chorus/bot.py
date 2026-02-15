"""Discord bot entrypoint for Chorus."""

from __future__ import annotations

import asyncio
import logging
import os
import pkgutil
from datetime import UTC, datetime
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
from chorus.config import BotConfig, GlobalConfig
from chorus.llm.discovery import OPENAI_CHAT_PREFIXES, get_cached_models, validate_and_discover
from chorus.llm.providers import AnthropicProvider, OpenAIProvider
from chorus.llm.tool_loop import (
    ToolExecutionContext,
    ToolLoopEvent,
    ToolLoopEventType,
    run_tool_loop,
)
from chorus.permissions.ask_ui import ask_user_permission
from chorus.permissions.engine import get_preset
from chorus.storage.db import Database
from chorus.tools.registry import create_default_registry
from chorus.ui.status import BotPresenceManager, LiveStatusView

if TYPE_CHECKING:
    from chorus.agent.message_queue import ChannelMessageQueue
    from chorus.models import Agent

logger = logging.getLogger("chorus.bot")


class ChorusBot(commands.Bot):
    """Discord bot that turns channels into autonomous AI agents."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.global_config = GlobalConfig()

        # Thread and context management state (initialized early so on_message works pre-setup_hook)
        self._thread_managers: dict[str, ThreadManager] = {}
        self._context_managers: dict[str, ContextManager] = {}
        self._channel_to_agent: dict[int, str] = {}
        self._message_queues: dict[int, ChannelMessageQueue] = {}
        self._test_channel: discord.TextChannel | None = None
        self._presence_manager: BotPresenceManager | None = None
        self._discovery_task: asyncio.Task[None] | None = None

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
        template_env = os.environ.get("CHORUS_TEMPLATE_DIR")
        if template_env:
            template_dir = Path(template_env)
        else:
            template_dir = Path(__file__).resolve().parent.parent.parent / "template"
        directory = AgentDirectory(self.config.chorus_home, template_dir)
        directory.ensure_home()
        self.global_config = GlobalConfig.load(self.config.chorus_home / "config.json")
        self.agent_manager = AgentManager(
            directory, self.db, global_config=self.global_config, bot_config=self.config
        )

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

        # Expose scope path to agent subprocesses
        if self.config.scope_path is not None:
            os.environ["SCOPE_PATH"] = str(self.config.scope_path)

        # Initialize presence manager
        self._presence_manager = BotPresenceManager(self)

        # Launch background model discovery
        self._discovery_task = asyncio.create_task(self._run_startup_discovery())

    async def _run_startup_discovery(self) -> None:
        """Run model discovery in the background on startup."""
        try:
            result = await asyncio.wait_for(
                validate_and_discover(
                    self.config.chorus_home,
                    anthropic_key=self.config.anthropic_api_key,
                    openai_key=self.config.openai_api_key,
                ),
                timeout=30,
            )
            providers = result.get("providers", {})
            total = sum(
                len(p.get("models", [])) for p in providers.values() if p.get("valid")
            )
            logger.info(
                "Startup discovery: %d models from %d providers", total, len(providers)
            )
        except TimeoutError:
            logger.warning("Startup model discovery timed out after 30s")
        except asyncio.CancelledError:
            logger.info("Startup model discovery cancelled")
        except Exception:
            logger.exception("Startup model discovery failed")

    async def close(self) -> None:
        """Kill all threads, cancel discovery, shut down database, and disconnect."""
        if self._discovery_task is not None and not self._discovery_task.done():
            self._discovery_task.cancel()
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

        await self._reconcile_channels()
        await self._ensure_test_channel()

    async def _reconcile_channels(self) -> None:
        """Reconcile Discord channels with agent records on startup."""
        if not self.config.dev_guild_id:
            return

        guild = self.get_guild(self.config.dev_guild_id)
        if guild is None:
            return

        category = discord.utils.get(guild.categories, name="Chorus Agents")
        if category is None:
            return  # No agent category yet — nothing to reconcile

        db_agents = await self.db.list_agents(guild.id)
        agent_channel_ids = {a["channel_id"] for a in db_agents}

        # Direction 1: Ghost channels → delete
        for channel in list(category.text_channels):
            if channel.id not in agent_channel_ids:
                logger.info("Deleting ghost channel #%s (id=%d)", channel.name, channel.id)
                await channel.delete(reason="Chorus reconciliation: no agent record")

        # Direction 2: Orphaned agents → recreate channel
        guild_channel_ids = {ch.id for ch in guild.text_channels}
        for agent_data in db_agents:
            if agent_data["channel_id"] not in guild_channel_ids:
                name = str(agent_data["name"])
                logger.info("Recreating channel for orphaned agent %s", name)
                new_channel = await guild.create_text_channel(name, category=category)
                await self.db.update_agent_channel(name, new_channel.id)
                self.agent_manager._directory.update_channel_id(name, new_channel.id)

    async def _ensure_test_channel(self) -> None:
        """Find or create the live test channel in the dev guild."""
        if not self.config.live_test_enabled or not self.config.dev_guild_id:
            return

        guild = self.get_guild(self.config.dev_guild_id)
        if guild is None:
            logger.warning(
                "Dev guild %s not found — skipping test channel", self.config.dev_guild_id
            )
            return

        channel_name = self.config.live_test_channel or "chorus-live-tests"

        # Find or create category
        category = discord.utils.get(guild.categories, name="Chorus Testing")
        if category is None:
            category = await guild.create_category("Chorus Testing")
            logger.info("Created 'Chorus Testing' category in guild %s", guild.name)

        # Find or create channel
        channel = discord.utils.get(guild.text_channels, name=channel_name, category=category)
        if channel is None:
            channel = await guild.create_text_channel(channel_name, category=category)
            logger.info("Created test channel #%s in guild %s", channel_name, guild.name)

        self._test_channel = channel

        assert self.user is not None
        await channel.send(
            f"**Chorus Bot Online**\n"
            f"Bot: {self.user.name} | Guilds: {len(self.guilds)} | "
            f"Latency: {self.latency * 1000:.0f}ms"
        )
        logger.info("Test channel ready: #%s (id=%d)", channel.name, channel.id)

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
        tm = self._thread_managers.setdefault(agent_name, ThreadManager(agent_name, db=self.db))

        # Ensure ThreadManager is initialized (loads latest branch_id from DB)
        await tm.initialize()

        # Get or create ContextManager for this agent
        if agent_name not in self._context_managers:
            agent_dir = self.agent_manager._directory._agents_dir / agent_name
            sessions_dir = agent_dir / "sessions"
            self._context_managers[agent_name] = ContextManager(agent_name, self.db, sessions_dir)
        cm = self._context_managers[agent_name]

        # ── Reply-based routing (preserved) ──────────────────────────────
        if message.reference and message.reference.message_id:
            thread = tm.route_message(message.reference.message_id)
            if thread is not None:
                thread.messages.append({"role": "user", "content": message.content})
                await cm.persist_message(
                    role="user",
                    content=message.content,
                    thread_id=thread.id,
                    discord_message_id=message.id,
                )
                # Update last_message_at for idle tracking
                await self._update_branch_last_message(agent_name, thread.id)
                if thread.status != ThreadStatus.RUNNING:
                    agent = await self.agent_manager.get_agent_by_channel(message.channel.id)
                    if agent is None:
                        return
                    _is_admin = (
                        hasattr(message.author, "guild_permissions")
                        and message.author.guild_permissions.manage_guild
                    )
                    runner = self._make_llm_runner(
                        agent, tm, cm,
                        channel=message.channel,
                        author_id=message.author.id,
                        is_admin=bool(_is_admin),
                        target_thread=thread,
                        reference=message,
                    )
                    tm.start_thread(thread, runner=runner)
                return

        # ── Non-reply: always create a new branch ─────────────────────────
        thread = await tm.create_branch({"role": "user", "content": message.content})
        await cm.persist_message(
            role="user",
            content=message.content,
            thread_id=thread.id,
            discord_message_id=message.id,
        )
        await self._update_branch_last_message(agent_name, thread.id)
        await self._generate_branch_summary(agent_name, thread.id, message.content)
        agent = await self.agent_manager.get_agent_by_channel(message.channel.id)
        if agent is None:
            return
        _is_admin = (
            hasattr(message.author, "guild_permissions")
            and message.author.guild_permissions.manage_guild
        )
        runner = self._make_llm_runner(
            agent, tm, cm,
            channel=message.channel,
            author_id=message.author.id,
            is_admin=bool(_is_admin),
            target_thread=thread,
            reference=message,
        )
        tm.start_thread(thread, runner=runner)

    async def _update_branch_last_message(self, agent_name: str, branch_id: int) -> None:
        """Update last_message_at for a branch in the DB."""
        try:
            await self.db.update_branch_last_message(
                agent_name, branch_id, datetime.now(UTC).isoformat()
            )
        except Exception:
            logger.debug("Failed to update branch last_message_at", exc_info=True)

    async def _generate_branch_summary(
        self, agent_name: str, branch_id: int, first_message: str
    ) -> None:
        """Set a short text summary for a branch from the first message."""
        summary = first_message[:50].strip()
        if len(first_message) > 50:
            summary += "..."
        try:
            await self.db.set_branch_summary(agent_name, branch_id, summary)
        except Exception:
            logger.debug("Failed to set branch summary", exc_info=True)

    def _make_llm_runner(
        self,
        agent: Agent,
        tm: ThreadManager,
        cm: ContextManager,
        channel: discord.abc.Messageable,
        author_id: int,
        is_admin: bool,
        target_thread: ExecutionThread,
        *,
        reference: discord.Message | None = None,
        model_override: str | None = None,
    ) -> ThreadRunner:
        """Build a ThreadRunner closure that runs the LLM tool loop."""

        async def runner(thread: ExecutionThread) -> None:
            # Resolve permission profile
            profile = get_preset(agent.permissions)

            # Choose provider based on model name
            model = (
                model_override
                or agent.model
                or self.global_config.default_model
                or "claude-sonnet-4-20250514"
            )
            provider: AnthropicProvider | OpenAIProvider
            if model.startswith(OPENAI_CHAT_PREFIXES):
                if not self.config.openai_api_key:
                    await channel.send("No OpenAI API key configured.")
                    return
                provider = OpenAIProvider(self.config.openai_api_key, model)
            else:
                if not self.config.anthropic_api_key:
                    await channel.send("No Anthropic API key configured.")
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
                chorus_home=self.config.chorus_home,
                is_admin=is_admin,
                db=self.db,
                host_execution=self.config.host_execution,
            )

            # Get previous branch summary for context
            previous_summary: str | None = None
            previous_branch_id: int | None = None
            if thread.id > 1:
                prev_id = thread.id - 1
                try:
                    prev_branch = await self.db.get_branch(agent.name, prev_id)
                    if prev_branch and prev_branch.get("summary"):
                        previous_summary = prev_branch["summary"]
                        previous_branch_id = prev_id
                except Exception:
                    pass

            # Get available models for self-awareness
            available_models = get_cached_models(self.config.chorus_home)

            llm_messages = await build_llm_context(
                agent=agent,
                thread_id=thread.id,
                context_manager=cm,
                thread_manager=tm,
                agent_docs_dir=docs_dir,
                model=model,
                available_models=available_models,
                previous_branch_summary=previous_summary,
                previous_branch_id=previous_branch_id,
                scope_path=self.config.scope_path,
            )

            registry = create_default_registry()

            # Set up live status view
            status_view = LiveStatusView(
                channel=channel,
                agent_name=agent.name,
                thread_id=thread.id,
                get_active_count=lambda: len(tm.list_active()),
                reference=reference,
            )
            await status_view.start()
            if status_view.message is not None:
                tm.register_bot_message(status_view.message.id, thread.id)
            if self._presence_manager:
                await self._presence_manager.thread_started(agent.name, thread.id)

            # Bridge tool loop events to status view
            recent_commands: list[str] = []

            async def on_event(event: ToolLoopEvent) -> None:
                updates: dict[str, object] = {}
                if event.total_usage:
                    updates["token_usage"] = event.total_usage
                updates["tool_calls_made"] = event.tool_calls_made
                updates["tools_used"] = list(event.tools_used)
                updates["llm_iterations"] = event.iteration
                if event.type == ToolLoopEventType.TOOL_CALL_START and event.tool_name:
                    updates["current_step"] = f"Running {event.tool_name}"
                    updates["step_number"] = event.tool_calls_made + 1
                    # Track recent bash/git commands for live status
                    if event.tool_arguments:
                        cmd: str | None = None
                        if event.tool_name == "bash":
                            cmd = event.tool_arguments.get("command")
                        elif event.tool_name and event.tool_name.startswith("git_"):
                            op = event.tool_name.removeprefix("git_")
                            args = " ".join(str(v) for v in event.tool_arguments.values())
                            cmd = f"git {op} {args}".strip()
                        if cmd:
                            recent_commands.append(cmd)
                            if len(recent_commands) > 5:
                                recent_commands.pop(0)
                            updates["recent_commands"] = list(recent_commands)
                elif event.type == ToolLoopEventType.LLM_CALL_START:
                    updates["current_step"] = f"Thinking (call {event.iteration})"
                await status_view.update(**updates)

            # Build ask_callback for ASK permission prompts
            async def _ask_callback(tool_name: str, arguments: str) -> bool:
                import json as _json

                from chorus.llm.tool_loop import _build_action_string
                try:
                    args = _json.loads(arguments)
                except (ValueError, TypeError):
                    args = {}
                action = _build_action_string(tool_name, args)
                return await ask_user_permission(
                    channel=channel,  # type: ignore[arg-type]
                    requester_id=author_id,
                    action_string=action,
                    tool_name=tool_name,
                    arguments=arguments,
                )

            try:
                # Run the tool loop
                result = await run_tool_loop(
                    provider=provider,
                    messages=llm_messages,
                    tools=registry,
                    ctx=ctx,
                    system_prompt=agent.system_prompt,
                    model=model,
                    max_iterations=self.global_config.max_tool_loop_iterations,
                    ask_callback=_ask_callback,
                    inject_queue=thread.inject_queue,
                    on_event=on_event,
                    web_search_enabled=agent.web_search and isinstance(provider, AnthropicProvider),
                )

                # Finalize status with response content
                await status_view.finalize(
                    "completed", response_content=result.content
                )

                # Register all chunk message IDs for reply-based routing
                for msg_id in status_view.chunk_message_ids:
                    tm.register_bot_message(msg_id, thread.id)

                # Persist assistant message using the last message ID
                if result.content and status_view.message is not None:
                    await cm.persist_message(
                        role="assistant",
                        content=result.content,
                        thread_id=thread.id,
                        discord_message_id=status_view.message.id,
                    )
            except Exception as exc:
                await status_view.finalize("error", error=str(exc))
                raise
            finally:
                if self._presence_manager:
                    await self._presence_manager.thread_completed(agent.name, thread.id)

        return runner

    async def run_model_query(
        self,
        channel: discord.abc.Messageable,
        author_id: int,
        is_admin: bool,
        prompt: str,
        model_id: str,
    ) -> None:
        """Run a one-off query with a specific model, using full agent context.

        Called by model shortcut slash commands (e.g. /haiku, /sonnet).
        Does NOT change the agent's default model.
        """
        channel_id = getattr(channel, "id", None)
        if channel_id is None:
            raise ValueError("Channel has no id attribute")

        # Look up agent
        agent = await self.agent_manager.get_agent_by_channel(channel_id)
        if agent is None:
            raise ValueError("No agent bound to this channel.")

        agent_name = agent.name

        # Cache channel→agent mapping
        self._channel_to_agent[channel_id] = agent_name

        # Get or create ThreadManager
        tm = self._thread_managers.setdefault(
            agent_name, ThreadManager(agent_name, db=self.db)
        )
        await tm.initialize()

        # Get or create ContextManager
        if agent_name not in self._context_managers:
            agent_dir = self.agent_manager._directory._agents_dir / agent_name
            sessions_dir = agent_dir / "sessions"
            self._context_managers[agent_name] = ContextManager(
                agent_name, self.db, sessions_dir
            )
        cm = self._context_managers[agent_name]

        # Create a new branch with the user message
        thread = await tm.create_branch({"role": "user", "content": prompt})
        await cm.persist_message(
            role="user",
            content=prompt,
            thread_id=thread.id,
            discord_message_id=None,
        )
        await self._update_branch_last_message(agent_name, thread.id)
        await self._generate_branch_summary(agent_name, thread.id, prompt)

        # Build runner with model override
        runner = self._make_llm_runner(
            agent, tm, cm,
            channel=channel,
            author_id=author_id,
            is_admin=is_admin,
            target_thread=thread,
            model_override=model_id,
        )
        tm.start_thread(thread, runner=runner)


def main() -> None:
    """Entry point: load env, build config, run bot."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    config = BotConfig.from_env()
    bot = ChorusBot(config)
    bot.run(config.discord_token, log_handler=None)
