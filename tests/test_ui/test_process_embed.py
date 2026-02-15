"""Tests for chorus.ui.process_embed — live-updating Discord embed."""

from __future__ import annotations

from collections import deque
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from chorus.process.models import (
    CallbackAction,
    HookTrigger,
    ProcessCallback,
    ProcessStatus,
    ProcessType,
    TriggerType,
)
from chorus.ui.process_embed import ProcessStatusEmbed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracked(**kwargs: Any) -> MagicMock:
    """Build a mock TrackedProcess for embed tests."""
    defaults: dict[str, Any] = {
        "pid": 100,
        "command": "python server.py",
        "status": ProcessStatus.RUNNING,
        "process_type": ProcessType.BACKGROUND,
        "exit_code": None,
        "rolling_tail": deque(),
        "callbacks": [],
    }
    defaults.update(kwargs)
    return MagicMock(**defaults)


def _make_embed(
    process_manager: Any | None = None,
    pid: int = 100,
    command: str = "python server.py",
    agent_name: str = "test-agent",
) -> ProcessStatusEmbed:
    """Build a ProcessStatusEmbed with mock channel and process manager."""
    channel = AsyncMock()
    pm = process_manager or MagicMock()
    return ProcessStatusEmbed(
        channel=channel,
        pid=pid,
        command=command,
        agent_name=agent_name,
        process_manager=pm,
    )


# ---------------------------------------------------------------------------
# Build embed tests
# ---------------------------------------------------------------------------


class TestBuildEmbed:
    def test_embed_title_contains_pid(self) -> None:
        pm = MagicMock()
        pm.get_process.return_value = _make_tracked(pid=100)
        embed_view = _make_embed(process_manager=pm, pid=100)
        embed = embed_view._build_embed()

        assert "100" in embed.title

    def test_embed_not_found(self) -> None:
        pm = MagicMock()
        pm.get_process.return_value = None
        embed_view = _make_embed(process_manager=pm)
        embed = embed_view._build_embed()

        assert "not found" in embed.description.lower()

    def test_embed_shows_command(self) -> None:
        pm = MagicMock()
        pm.get_process.return_value = _make_tracked()
        embed_view = _make_embed(process_manager=pm, command="python server.py")
        embed = embed_view._build_embed()

        fields = {f.name: f.value for f in embed.fields}
        assert "python server.py" in fields["Command"]

    def test_embed_truncates_long_command(self) -> None:
        long_cmd = "python " + "x" * 100
        pm = MagicMock()
        pm.get_process.return_value = _make_tracked(command=long_cmd)
        embed_view = _make_embed(process_manager=pm, command=long_cmd)
        embed = embed_view._build_embed()

        fields = {f.name: f.value for f in embed.fields}
        assert "..." in fields["Command"]

    def test_embed_green_when_running(self) -> None:
        pm = MagicMock()
        pm.get_process.return_value = _make_tracked(status=ProcessStatus.RUNNING)
        embed_view = _make_embed(process_manager=pm)
        embed = embed_view._build_embed()

        assert embed.color == discord.Color.green()

    def test_embed_red_on_error_exit(self) -> None:
        pm = MagicMock()
        pm.get_process.return_value = _make_tracked(
            status=ProcessStatus.EXITED,
            exit_code=1,
        )
        embed_view = _make_embed(process_manager=pm)
        embed = embed_view._build_embed()

        assert embed.color == discord.Color.red()

    def test_embed_grey_on_clean_exit(self) -> None:
        pm = MagicMock()
        pm.get_process.return_value = _make_tracked(
            status=ProcessStatus.EXITED,
            exit_code=0,
        )
        embed_view = _make_embed(process_manager=pm)
        embed = embed_view._build_embed()

        assert embed.color == discord.Color.greyple()

    def test_embed_shows_exit_code(self) -> None:
        pm = MagicMock()
        pm.get_process.return_value = _make_tracked(
            status=ProcessStatus.EXITED,
            exit_code=137,
        )
        embed_view = _make_embed(process_manager=pm)
        embed = embed_view._build_embed()

        fields = {f.name: f.value for f in embed.fields}
        assert "137" in fields["Exit Code"]

    def test_embed_shows_recent_output(self) -> None:
        pm = MagicMock()
        pm.get_process.return_value = _make_tracked(
            rolling_tail=deque(["line 1", "line 2", "line 3"]),
        )
        embed_view = _make_embed(process_manager=pm)
        embed = embed_view._build_embed()

        fields = {f.name: f.value for f in embed.fields}
        assert "line 1" in fields["Recent Output"]
        assert "line 3" in fields["Recent Output"]

    def test_embed_shows_agent_in_footer(self) -> None:
        pm = MagicMock()
        pm.get_process.return_value = _make_tracked()
        embed_view = _make_embed(process_manager=pm, agent_name="my-agent")
        embed = embed_view._build_embed()

        assert "my-agent" in embed.footer.text

    def test_embed_shows_active_watchers(self) -> None:
        cb = MagicMock(spec=ProcessCallback)
        cb.exhausted = False
        cb.trigger = MagicMock(spec=HookTrigger)
        cb.trigger.type = TriggerType.ON_OUTPUT_MATCH
        cb.trigger.pattern = "error"
        cb.action = CallbackAction.SPAWN_BRANCH

        pm = MagicMock()
        pm.get_process.return_value = _make_tracked(callbacks=[cb])
        embed_view = _make_embed(process_manager=pm)
        embed = embed_view._build_embed()

        fields = {f.name: f.value for f in embed.fields}
        assert "Active Watchers" in fields
        assert "error" in fields["Active Watchers"]

    def test_embed_shows_exit_type_watchers(self) -> None:
        cb = MagicMock(spec=ProcessCallback)
        cb.exhausted = False
        cb.trigger = MagicMock(spec=HookTrigger)
        cb.trigger.type = TriggerType.ON_EXIT
        cb.trigger.pattern = None
        cb.action = CallbackAction.SPAWN_BRANCH

        pm = MagicMock()
        pm.get_process.return_value = _make_tracked(callbacks=[cb])
        embed_view = _make_embed(process_manager=pm)
        embed = embed_view._build_embed()

        fields = {f.name: f.value for f in embed.fields}
        assert "Active Watchers" in fields
        assert "on_exit" in fields["Active Watchers"]


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------


class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_sends_embed(self) -> None:
        pm = MagicMock()
        pm.get_process.return_value = _make_tracked()
        embed_view = _make_embed(process_manager=pm)

        msg = await embed_view.start()

        embed_view._channel.send.assert_called_once()
        assert msg is not None

    @pytest.mark.asyncio
    async def test_start_returns_none_on_send_failure(self) -> None:
        pm = MagicMock()
        pm.get_process.return_value = _make_tracked()
        embed_view = _make_embed(process_manager=pm)
        embed_view._channel.send.side_effect = discord.HTTPException(
            MagicMock(status=500), "error"
        )

        msg = await embed_view.start()

        assert msg is None

    @pytest.mark.asyncio
    async def test_stop_cancels_ticker(self) -> None:
        pm = MagicMock()
        pm.get_process.return_value = _make_tracked()
        embed_view = _make_embed(process_manager=pm)

        await embed_view.start()
        assert embed_view._ticker_task is not None

        await embed_view.stop()
        assert embed_view._ticker_task.done()

    @pytest.mark.asyncio
    async def test_stop_does_final_update(self) -> None:
        pm = MagicMock()
        pm.get_process.return_value = _make_tracked(
            status=ProcessStatus.EXITED, exit_code=0
        )
        embed_view = _make_embed(process_manager=pm)

        await embed_view.start()
        await embed_view.stop()

        # The message should have been edited at least once for final update
        assert embed_view._message.edit.called


# ---------------------------------------------------------------------------
# Uptime formatting
# ---------------------------------------------------------------------------


class TestUptimeFormatting:
    def test_uptime_seconds(self) -> None:
        pm = MagicMock()
        pm.get_process.return_value = _make_tracked()
        embed_view = _make_embed(process_manager=pm)
        # _started_at is set to now, so uptime should be < 1s
        embed = embed_view._build_embed()

        fields = {f.name: f.value for f in embed.fields}
        assert "s" in fields["Uptime"]
