"""Tests for /process slash commands."""

from __future__ import annotations

from collections import deque
from unittest.mock import AsyncMock, MagicMock

import pytest

from chorus.commands.process_commands import ProcessCog
from chorus.process.models import (
    CallbackAction,
    ExitFilter,
    HookTrigger,
    ProcessCallback,
    ProcessStatus,
    ProcessType,
    TrackedProcess,
    TriggerType,
)


def _make_tracked(**overrides: object) -> TrackedProcess:
    defaults: dict[str, object] = {
        "pid": 12345,
        "command": "python server.py",
        "working_directory": "/tmp/workspace",
        "agent_name": "test-agent",
        "started_at": "2026-02-14T12:00:00+00:00",
        "process_type": ProcessType.BACKGROUND,
        "status": ProcessStatus.RUNNING,
    }
    defaults.update(overrides)
    return TrackedProcess(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def mock_bot() -> MagicMock:
    bot = MagicMock()
    bot._channel_to_agent = {123: "test-agent"}
    bot._process_manager = MagicMock()
    return bot


@pytest.fixture
def cog(mock_bot: MagicMock) -> ProcessCog:
    return ProcessCog(mock_bot)


@pytest.fixture
def interaction() -> MagicMock:
    inter = MagicMock()
    inter.channel = MagicMock()
    inter.channel.id = 123
    inter.response = AsyncMock()
    inter.followup = AsyncMock()
    return inter


@pytest.mark.asyncio
async def test_process_list_empty(cog: ProcessCog, interaction: MagicMock) -> None:
    """Shows ephemeral message when no processes."""
    cog.bot._process_manager.list_processes.return_value = []
    await cog.process_list.callback(cog, interaction)
    interaction.response.send_message.assert_awaited_once()
    assert "No processes" in str(interaction.response.send_message.call_args)


@pytest.mark.asyncio
async def test_process_list_with_processes(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """Shows embed with process info."""
    p = _make_tracked()
    cog.bot._process_manager.list_processes.return_value = [p]
    await cog.process_list.callback(cog, interaction)
    interaction.response.send_message.assert_awaited_once()
    call_kwargs = interaction.response.send_message.call_args[1]
    embed = call_kwargs.get("embed")
    assert embed is not None
    assert "12345" in str(embed.fields[0].name)


@pytest.mark.asyncio
async def test_process_kill_success(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """Kill command kills a process."""
    p = _make_tracked()
    cog.bot._process_manager.get_process.return_value = p
    cog.bot._process_manager.kill_process = AsyncMock(return_value=True)
    await cog.process_kill.callback(cog, interaction, pid=12345)
    cog.bot._process_manager.kill_process.assert_awaited_with(12345)
    assert "Killed" in interaction.response.send_message.call_args[0][0]


@pytest.mark.asyncio
async def test_process_kill_not_found(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """Kill command reports error for unknown PID."""
    cog.bot._process_manager.get_process.return_value = None
    await cog.process_kill.callback(cog, interaction, pid=99999)
    assert "found" in str(interaction.response.send_message.call_args).lower()


@pytest.mark.asyncio
async def test_process_kill_wrong_agent(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """Kill command rejects process from different agent."""
    p = _make_tracked(agent_name="other-agent")
    cog.bot._process_manager.get_process.return_value = p
    await cog.process_kill.callback(cog, interaction, pid=12345)
    assert "different agent" in str(interaction.response.send_message.call_args).lower()


@pytest.mark.asyncio
async def test_process_logs(cog: ProcessCog, interaction: MagicMock) -> None:
    """Logs command shows rolling tail."""
    p = _make_tracked()
    p.rolling_tail = deque(["line 1", "line 2", "line 3"], maxlen=100)
    cog.bot._process_manager.get_process.return_value = p
    await cog.process_logs.callback(cog, interaction, pid=12345, lines=20)
    msg = interaction.response.send_message.call_args[0][0]
    assert "line 1" in msg
    assert "line 2" in msg
    assert "line 3" in msg


@pytest.mark.asyncio
async def test_process_logs_empty(cog: ProcessCog, interaction: MagicMock) -> None:
    """Logs command shows message when no output."""
    p = _make_tracked()
    cog.bot._process_manager.get_process.return_value = p
    await cog.process_logs.callback(cog, interaction, pid=12345, lines=20)
    assert "No output" in str(interaction.response.send_message.call_args)


@pytest.mark.asyncio
async def test_process_list_full_command_visible(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """Process list shows full command, not truncated to 60 chars."""
    long_cmd = "python3 /very/long/path/to/some/script.py --with-lots-of-arguments --verbose --debug"
    p = _make_tracked(command=long_cmd)
    cog.bot._process_manager.list_processes.return_value = [p]
    await cog.process_list.callback(cog, interaction)
    call_kwargs = interaction.response.send_message.call_args[1]
    embed = call_kwargs["embed"]
    field_value = embed.fields[0].value
    # Full command should appear, not truncated to 60 chars
    assert long_cmd in field_value


@pytest.mark.asyncio
async def test_process_list_shows_callbacks(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """Process list shows active callbacks."""
    cb = ProcessCallback(
        trigger=HookTrigger(type=TriggerType.ON_EXIT, exit_filter=ExitFilter.FAILURE),
        action=CallbackAction.NOTIFY_CHANNEL,
    )
    p = _make_tracked(callbacks=[cb], context="watch for failures")
    cog.bot._process_manager.list_processes.return_value = [p]
    await cog.process_list.callback(cog, interaction)
    call_kwargs = interaction.response.send_message.call_args[1]
    embed = call_kwargs["embed"]
    field_value = embed.fields[0].value
    assert "notify_channel" in field_value
    assert "on_exit" in field_value
    assert "watch for failures" in field_value


@pytest.mark.asyncio
async def test_process_list_no_manager(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """Commands handle missing process manager gracefully."""
    cog.bot._process_manager = None
    await cog.process_list.callback(cog, interaction)
    assert "not available" in str(interaction.response.send_message.call_args).lower()
