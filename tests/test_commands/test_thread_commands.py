"""Tests for chorus.commands.thread_commands — /branch slash commands."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from chorus.agent.threads import ExecutionThread, ThreadManager, ThreadStatus
from chorus.commands.thread_commands import ThreadCog


def _make_interaction(bot: MagicMock, channel_id: int = 987654321) -> MagicMock:
    interaction = MagicMock()
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.channel = MagicMock()
    interaction.channel.id = channel_id
    interaction.guild_id = 123456789
    return interaction


class TestBranchList:
    async def test_list_shows_branches(self, mock_bot: MagicMock) -> None:
        tm = ThreadManager("test-agent")
        t1 = tm.create_thread({"role": "user", "content": "hello"})
        t1.summary = "Auth refactor"
        t1.metrics.begin_step("Calling LLM")

        mock_bot._thread_managers = {"test-agent": tm}
        mock_bot._channel_to_agent = {987654321: "test-agent"}

        cog = ThreadCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.branch_list.callback(cog, interaction)

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args
        embed = call_kwargs.kwargs.get("embed")
        assert embed is not None

    async def test_list_empty(self, mock_bot: MagicMock) -> None:
        mock_bot._thread_managers = {}
        mock_bot._channel_to_agent = {}

        cog = ThreadCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.branch_list.callback(cog, interaction)

        interaction.response.send_message.assert_called_once()


class TestBranchKill:
    async def test_kill_by_id(self, mock_bot: MagicMock) -> None:
        tm = ThreadManager("test-agent")
        thread = tm.create_thread({"role": "user", "content": "hello"})

        async def slow(t: ExecutionThread) -> None:
            import asyncio

            await asyncio.sleep(100)

        tm.start_thread(thread, runner=slow)

        mock_bot._thread_managers = {"test-agent": tm}
        mock_bot._channel_to_agent = {987654321: "test-agent"}

        cog = ThreadCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.branch_kill.callback(cog, interaction, target="1")

        assert thread.status == ThreadStatus.COMPLETED
        interaction.response.send_message.assert_called_once()

    async def test_kill_all(self, mock_bot: MagicMock) -> None:
        tm = ThreadManager("test-agent")

        async def slow(t: ExecutionThread) -> None:
            import asyncio

            await asyncio.sleep(100)

        t1 = tm.create_thread({"role": "user", "content": "a"})
        t2 = tm.create_thread({"role": "user", "content": "b"})
        tm.start_thread(t1, runner=slow)
        tm.start_thread(t2, runner=slow)

        mock_bot._thread_managers = {"test-agent": tm}
        mock_bot._channel_to_agent = {987654321: "test-agent"}

        cog = ThreadCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.branch_kill.callback(cog, interaction, target="all")

        assert t1.status == ThreadStatus.COMPLETED
        assert t2.status == ThreadStatus.COMPLETED

    async def test_kill_nonexistent(self, mock_bot: MagicMock) -> None:
        tm = ThreadManager("test-agent")
        mock_bot._thread_managers = {"test-agent": tm}
        mock_bot._channel_to_agent = {987654321: "test-agent"}

        cog = ThreadCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.branch_kill.callback(cog, interaction, target="999")

        interaction.response.send_message.assert_called_once()
        msg = interaction.response.send_message.call_args[0][0]
        assert "not found" in msg.lower() or "no branch" in msg.lower()


class TestBreakContext:
    async def test_break_context_detaches_main(self, mock_bot: MagicMock) -> None:
        tm = ThreadManager("test-agent")
        thread = tm.create_thread({"role": "user", "content": "hello"}, is_main=True)
        assert tm.get_main_thread() is thread

        mock_bot._thread_managers = {"test-agent": tm}
        mock_bot._channel_to_agent = {987654321: "test-agent"}

        cog = ThreadCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.break_context.callback(cog, interaction)

        assert tm.get_main_thread() is None
        interaction.response.send_message.assert_called_once()

    async def test_break_context_no_agent(self, mock_bot: MagicMock) -> None:
        mock_bot._thread_managers = {}
        mock_bot._channel_to_agent = {}

        cog = ThreadCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.break_context.callback(cog, interaction)

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args
        assert call_kwargs.kwargs.get("ephemeral") is True

    async def test_break_context_no_main_branch(self, mock_bot: MagicMock) -> None:
        tm = ThreadManager("test-agent")
        mock_bot._thread_managers = {"test-agent": tm}
        mock_bot._channel_to_agent = {987654321: "test-agent"}

        cog = ThreadCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.break_context.callback(cog, interaction)

        interaction.response.send_message.assert_called_once()
        msg = interaction.response.send_message.call_args[0][0]
        assert "no main branch" in msg.lower() or "no active" in msg.lower()

    async def test_break_context_response_mentions_old_branch(
        self, mock_bot: MagicMock
    ) -> None:
        tm = ThreadManager("test-agent")
        thread = tm.create_thread({"role": "user", "content": "hello"}, is_main=True)

        mock_bot._thread_managers = {"test-agent": tm}
        mock_bot._channel_to_agent = {987654321: "test-agent"}

        cog = ThreadCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.break_context.callback(cog, interaction)

        msg = interaction.response.send_message.call_args[0][0]
        assert str(thread.id) in msg

    async def test_break_context_running_branch_continues(
        self, mock_bot: MagicMock
    ) -> None:
        import asyncio

        tm = ThreadManager("test-agent")
        thread = tm.create_thread({"role": "user", "content": "hello"}, is_main=True)

        async def slow(t: ExecutionThread) -> None:
            await asyncio.sleep(100)

        tm.start_thread(thread, runner=slow)

        mock_bot._thread_managers = {"test-agent": tm}
        mock_bot._channel_to_agent = {987654321: "test-agent"}

        cog = ThreadCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.break_context.callback(cog, interaction)

        # Main branch detached but still running
        assert tm.get_main_thread() is None
        assert thread.status == ThreadStatus.RUNNING
        await tm.kill_all()


class TestBranchHistory:
    async def test_history_shows_steps(self, mock_bot: MagicMock) -> None:
        tm = ThreadManager("test-agent")
        thread = tm.create_thread({"role": "user", "content": "hello"})
        thread.metrics.begin_step("Step one")
        thread.metrics.begin_step("Step two")
        thread.metrics.finalize()

        mock_bot._thread_managers = {"test-agent": tm}
        mock_bot._channel_to_agent = {987654321: "test-agent"}

        cog = ThreadCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.branch_history.callback(cog, interaction, branch_id=1)

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args
        embed = call_kwargs.kwargs.get("embed")
        assert embed is not None
