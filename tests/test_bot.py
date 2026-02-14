"""Tests for chorus.bot — Discord bot setup and event handling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from discord import app_commands

from chorus.agent.directory import AgentDirectory
from chorus.agent.manager import AgentManager
from chorus.agent.threads import ExecutionThread, ThreadManager
from chorus.bot import ChorusBot
from chorus.config import BotConfig
from chorus.llm.providers import LLMResponse, Usage
from chorus.storage.db import Database

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


@pytest.fixture
def bot_config(tmp_chorus_home: Path) -> BotConfig:
    return BotConfig(
        discord_token="test-token",
        chorus_home=tmp_chorus_home,
    )


@pytest.fixture
def bot(bot_config: BotConfig) -> ChorusBot:
    return ChorusBot(bot_config)


@pytest.fixture
async def reconcile_db(tmp_chorus_home: Path) -> AsyncGenerator[Database, None]:
    """Database for reconciliation tests."""
    db = Database(tmp_chorus_home / "db" / "chorus.db")
    await db.init()
    yield db
    await db.close()


class TestBotSetup:
    def test_bot_creates_with_correct_intents(self, bot: ChorusBot) -> None:
        intents = bot.intents
        assert intents.guilds is True
        assert intents.guild_messages is True
        assert intents.message_content is True
        assert intents.presences is False
        assert intents.members is False

    async def test_bot_loads_cogs_from_commands_package(self, bot: ChorusBot) -> None:
        with patch.object(bot, "load_extension", new_callable=AsyncMock) as mock_load:
            await bot.setup_hook()
            # Should load at least agent_commands
            loaded = [call.args[0] for call in mock_load.call_args_list]
            assert "chorus.commands.agent_commands" in loaded


class TestBotEvents:
    async def test_bot_on_message_ignores_self(self, bot: ChorusBot) -> None:
        # Set up bot.user
        bot._connection.user = MagicMock(spec=discord.User)
        bot._connection.user.id = 444555666

        message = MagicMock(spec=discord.Message)
        message.author = bot._connection.user

        with patch.object(bot, "process_commands", new_callable=AsyncMock) as mock_process:
            await bot.on_message(message)
            mock_process.assert_not_called()

    async def test_bot_on_message_processes_others(self, bot: ChorusBot) -> None:
        bot._connection.user = MagicMock(spec=discord.User)
        bot._connection.user.id = 444555666

        message = MagicMock(spec=discord.Message)
        other_user = MagicMock(spec=discord.User)
        other_user.id = 999999
        message.author = other_user

        with patch.object(bot, "process_commands", new_callable=AsyncMock) as mock_process:
            await bot.on_message(message)
            mock_process.assert_called_once_with(message)


class TestBotErrorHandler:
    async def test_bot_error_handler_sends_ephemeral(self, bot: ChorusBot) -> None:
        with patch.object(bot, "load_extension", new_callable=AsyncMock):
            await bot.setup_hook()

        interaction = MagicMock(spec=discord.Interaction)
        interaction.response = MagicMock()
        interaction.response.is_done.return_value = False
        interaction.response.send_message = AsyncMock()
        interaction.followup = AsyncMock()

        error = app_commands.MissingPermissions(["admin"])

        await bot.tree.on_error(interaction, error)
        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args
        assert call_kwargs.kwargs.get("ephemeral") is True


class TestReconcileChannels:
    """Tests for _reconcile_channels() startup reconciliation."""

    def _make_mock_channel(self, channel_id: int, name: str) -> MagicMock:
        ch = MagicMock(spec=discord.TextChannel)
        ch.id = channel_id
        ch.name = name
        ch.delete = AsyncMock()
        return ch

    def _make_bot_with_guild(
        self,
        bot_config: BotConfig,
        guild_id: int,
        category_channels: list[MagicMock],
        all_text_channels: list[MagicMock] | None = None,
    ) -> ChorusBot:
        config = BotConfig(
            discord_token=bot_config.discord_token,
            chorus_home=bot_config.chorus_home,
            dev_guild_id=guild_id,
        )
        bot = ChorusBot(config)

        # Mock category
        category = MagicMock()
        category.name = "Chorus Agents"
        category.text_channels = category_channels

        # Mock guild
        guild = MagicMock(spec=discord.Guild)
        guild.id = guild_id
        guild.categories = [category]
        all_channels = all_text_channels if all_text_channels is not None else category_channels
        guild.text_channels = all_channels
        guild.create_text_channel = AsyncMock()

        bot.get_guild = MagicMock(return_value=guild)
        return bot

    async def test_reconcile_skips_without_dev_guild(
        self, bot_config: BotConfig, reconcile_db: Database
    ) -> None:
        """No dev_guild_id → no-op."""
        bot = ChorusBot(bot_config)  # no dev_guild_id
        bot.db = reconcile_db
        # Should return without error
        await bot._reconcile_channels()

    async def test_reconcile_skips_without_category(
        self, bot_config: BotConfig, reconcile_db: Database
    ) -> None:
        """No 'Chorus Agents' category → no-op."""
        config = BotConfig(
            discord_token="test-token",
            chorus_home=bot_config.chorus_home,
            dev_guild_id=123,
        )
        bot = ChorusBot(config)
        bot.db = reconcile_db

        guild = MagicMock(spec=discord.Guild)
        guild.id = 123
        guild.categories = []  # No categories
        bot.get_guild = MagicMock(return_value=guild)

        await bot._reconcile_channels()
        # No error = pass

    async def test_reconcile_deletes_ghost_channels(
        self, bot_config: BotConfig, reconcile_db: Database
    ) -> None:
        """Channel in category with no DB agent → deleted."""
        ghost_channel = self._make_mock_channel(111, "ghost-agent")
        bot = self._make_bot_with_guild(bot_config, 999, [ghost_channel])
        bot.db = reconcile_db

        await bot._reconcile_channels()

        ghost_channel.delete.assert_called_once_with(
            reason="Chorus reconciliation: no agent record"
        )

    async def test_reconcile_recreates_missing_channels(
        self,
        bot_config: BotConfig,
        reconcile_db: Database,
        tmp_chorus_home: Path,
        tmp_template: Path,
    ) -> None:
        """DB agent with no Discord channel → channel created, IDs updated."""
        # Register an agent in DB pointing to a channel that doesn't exist
        await reconcile_db.register_agent(
            name="orphan-agent",
            channel_id=555,
            guild_id=999,
            model=None,
            permissions="standard",
            created_at="2026-01-01T00:00:00",
        )

        # Create the agent directory so update_channel_id works
        directory = AgentDirectory(tmp_chorus_home, tmp_template)
        directory.create("orphan-agent")

        # No channels exist in Discord (empty list for category + guild)
        bot = self._make_bot_with_guild(bot_config, 999, [], [])
        bot.db = reconcile_db
        bot.agent_manager = AgentManager(directory, reconcile_db)

        # Mock the new channel that will be created
        new_channel = self._make_mock_channel(777, "orphan-agent")
        guild = bot.get_guild(999)
        guild.create_text_channel = AsyncMock(return_value=new_channel)

        await bot._reconcile_channels()

        guild.create_text_channel.assert_called_once()
        # DB should have the new channel_id
        agent = await reconcile_db.get_agent("orphan-agent")
        assert agent is not None
        assert agent["channel_id"] == 777

    async def test_reconcile_no_changes_when_synced(
        self, bot_config: BotConfig, reconcile_db: Database
    ) -> None:
        """Everything matches → nothing happens."""
        # Register an agent in DB
        await reconcile_db.register_agent(
            name="synced-agent",
            channel_id=222,
            guild_id=999,
            model=None,
            permissions="standard",
            created_at="2026-01-01T00:00:00",
        )

        # Channel exists in Discord with matching ID
        synced_channel = self._make_mock_channel(222, "synced-agent")
        bot = self._make_bot_with_guild(bot_config, 999, [synced_channel], [synced_channel])
        bot.db = reconcile_db

        await bot._reconcile_channels()

        # No deletions, no creations
        synced_channel.delete.assert_not_called()
        guild = bot.get_guild(999)
        guild.create_text_channel.assert_not_called()


# ---------------------------------------------------------------------------
# Live status feedback wiring
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal fake LLM provider for runner integration tests."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._idx = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        if self._idx >= len(self._responses):
            return LLMResponse(
                content="(exhausted)",
                tool_calls=[],
                stop_reason="end_turn",
                usage=Usage(0, 0),
                model="fake",
            )
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


class TestLiveStatusWiring:
    """Verify that _make_llm_runner wires status embed + presence correctly."""

    def _setup(
        self,
        bot_config: BotConfig,
        tmp_chorus_home: Path,
        tmp_template: Path,
        provider_responses: list[LLMResponse],
    ) -> tuple[ChorusBot, MagicMock, ExecutionThread]:
        """Build a ChorusBot with mocked components for runner tests."""
        from chorus.models import Agent

        bot_cfg = BotConfig(
            discord_token="test-token",
            chorus_home=tmp_chorus_home,
            anthropic_api_key="sk-ant-fake",
        )
        bot = ChorusBot(bot_cfg)

        # Presence manager
        bot._presence_manager = MagicMock()
        bot._presence_manager.thread_started = AsyncMock()
        bot._presence_manager.thread_completed = AsyncMock()

        # Agent directory
        directory = AgentDirectory(tmp_chorus_home, tmp_template)
        directory.ensure_home()
        directory.create("test-agent")

        bot.agent_manager = MagicMock()
        bot.agent_manager._directory._agents_dir = tmp_chorus_home / "agents"
        bot.global_config.max_tool_loop_iterations = 25
        bot.db = MagicMock()
        bot.db.get_branch = AsyncMock(return_value=None)

        # Mock channel + message
        mock_channel = MagicMock(spec=discord.TextChannel)
        mock_status_msg = MagicMock(spec=discord.Message)
        mock_status_msg.edit = AsyncMock()
        # The first send returns the status embed message, the second is the response
        bot_response_msg = MagicMock(spec=discord.Message)
        bot_response_msg.id = 99999
        mock_channel.send = AsyncMock(side_effect=[mock_status_msg, bot_response_msg])

        mock_message = MagicMock(spec=discord.Message)
        mock_message.channel = mock_channel
        mock_message.id = 12345
        mock_author = MagicMock()
        mock_author.guild_permissions = MagicMock()
        mock_author.guild_permissions.manage_guild = False
        mock_message.author = mock_author

        # Thread manager
        tm = ThreadManager("test-agent")
        thread = tm.create_thread({"role": "user", "content": "Hello"}, is_main=True)

        # Context manager mock — needs AsyncMock for async methods
        cm = MagicMock()
        cm.persist_message = AsyncMock()
        cm.get_context = AsyncMock(return_value=[])

        # Agent model
        agent = Agent(
            name="test-agent",
            channel_id=100,
            model=None,
            system_prompt="You are helpful.",
            permissions="open",
        )

        fake_provider = _FakeProvider(provider_responses)
        runner = bot._make_llm_runner(
            agent, tm, cm,
            channel=mock_channel,
            author_id=mock_message.author.id,
            is_admin=False,
            target_thread=thread,
            reference=mock_message,
        )

        # Stash refs for assertions
        bot._test_refs = {  # type: ignore[attr-defined]
            "channel": mock_channel,
            "status_msg": mock_status_msg,
            "message": mock_message,
            "tm": tm,
            "cm": cm,
            "thread": thread,
            "agent": agent,
            "provider": fake_provider,
            "runner": runner,
        }

        return bot, mock_channel, thread

    async def test_runner_sends_thinking_message(
        self, bot_config: BotConfig, tmp_chorus_home: Path, tmp_template: Path
    ) -> None:
        """Runner should send a plain 'Thinking...' message first."""
        responses = [
            LLMResponse(
                content="Hello!",
                tool_calls=[],
                stop_reason="end_turn",
                usage=Usage(10, 5),
                model="fake",
            ),
        ]
        bot, channel, thread = self._setup(bot_config, tmp_chorus_home, tmp_template, responses)
        runner = bot._test_refs["runner"]  # type: ignore[attr-defined]
        provider = bot._test_refs["provider"]  # type: ignore[attr-defined]

        with patch("chorus.bot.AnthropicProvider", return_value=provider):
            await runner(thread)

        # Channel.send called at least once (thinking message)
        assert channel.send.call_count >= 1
        first_call = channel.send.call_args_list[0]
        assert first_call.kwargs["content"] == "Thinking..."

    async def test_runner_finalizes_status_on_completion(
        self, bot_config: BotConfig, tmp_chorus_home: Path, tmp_template: Path
    ) -> None:
        """Status message is edited with response content after successful loop."""
        responses = [
            LLMResponse(
                content="Done!",
                tool_calls=[],
                stop_reason="end_turn",
                usage=Usage(10, 5),
                model="fake",
            ),
        ]
        bot, channel, thread = self._setup(bot_config, tmp_chorus_home, tmp_template, responses)
        runner = bot._test_refs["runner"]  # type: ignore[attr-defined]
        provider = bot._test_refs["provider"]  # type: ignore[attr-defined]

        with patch("chorus.bot.AnthropicProvider", return_value=provider):
            await runner(thread)

        # Status message should have been edited (finalize calls edit for <15s)
        status_msg = bot._test_refs["status_msg"]  # type: ignore[attr-defined]
        status_msg.edit.assert_called()
        # Last edit should contain the response text as plain content
        last_edit = status_msg.edit.call_args
        assert "Done!" in last_edit.kwargs["content"]

    async def test_runner_calls_presence_manager(
        self, bot_config: BotConfig, tmp_chorus_home: Path, tmp_template: Path
    ) -> None:
        """Presence manager thread_started/thread_completed are called."""
        responses = [
            LLMResponse(
                content="Hi!",
                tool_calls=[],
                stop_reason="end_turn",
                usage=Usage(10, 5),
                model="fake",
            ),
        ]
        bot, channel, thread = self._setup(bot_config, tmp_chorus_home, tmp_template, responses)
        runner = bot._test_refs["runner"]  # type: ignore[attr-defined]
        provider = bot._test_refs["provider"]  # type: ignore[attr-defined]

        with patch("chorus.bot.AnthropicProvider", return_value=provider):
            await runner(thread)

        bot._presence_manager.thread_started.assert_called_once()
        bot._presence_manager.thread_completed.assert_called_once()

    async def test_runner_includes_thread_id_in_finalized_embed(
        self, bot_config: BotConfig, tmp_chorus_home: Path, tmp_template: Path
    ) -> None:
        """Finalized embed footer should include branch ID."""
        responses = [
            LLMResponse(
                content="Hello!",
                tool_calls=[],
                stop_reason="end_turn",
                usage=Usage(10, 5),
                model="fake",
            ),
        ]
        bot, channel, thread = self._setup(bot_config, tmp_chorus_home, tmp_template, responses)
        runner = bot._test_refs["runner"]  # type: ignore[attr-defined]
        provider = bot._test_refs["provider"]  # type: ignore[attr-defined]

        with patch("chorus.bot.AnthropicProvider", return_value=provider):
            await runner(thread)

        # Finalized message content should have branch ID in footer
        status_msg = bot._test_refs["status_msg"]  # type: ignore[attr-defined]
        last_edit = status_msg.edit.call_args
        content = last_edit.kwargs["content"]
        assert f"branch #{thread.id}" in content
