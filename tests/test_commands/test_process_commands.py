"""Tests for /process slash commands."""

from __future__ import annotations

from collections import deque
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from chorus.commands.process_commands import (
    ProcessCog,
    _build_overflow_embed,
    _build_process_embed,
    _discord_timestamp,
    _format_hook,
    _safe_truncate,
)
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


# ---------------------------------------------------------------------------
# process_list — basic behavior
# ---------------------------------------------------------------------------


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
    """Shows embeds= list with one embed per process."""
    p = _make_tracked()
    cog.bot._process_manager.list_processes.return_value = [p]
    await cog.process_list.callback(cog, interaction)
    interaction.response.send_message.assert_awaited_once()
    call_kwargs = interaction.response.send_message.call_args[1]
    embeds = call_kwargs.get("embeds")
    assert embeds is not None
    assert len(embeds) == 1
    assert "12345" in embeds[0].title


@pytest.mark.asyncio
async def test_process_list_full_command_visible(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """Process list shows full command in embed description, not truncated."""
    long_cmd = "python3 /very/long/path/to/some/script.py --with-lots-of-arguments --verbose --debug"
    p = _make_tracked(command=long_cmd)
    cog.bot._process_manager.list_processes.return_value = [p]
    await cog.process_list.callback(cog, interaction)
    call_kwargs = interaction.response.send_message.call_args[1]
    embeds = call_kwargs["embeds"]
    # Full command in description (code block)
    assert long_cmd in embeds[0].description


@pytest.mark.asyncio
async def test_process_list_shows_callbacks(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """Process list shows hooks with trigger details."""
    cb = ProcessCallback(
        trigger=HookTrigger(type=TriggerType.ON_EXIT, exit_filter=ExitFilter.FAILURE),
        action=CallbackAction.NOTIFY_CHANNEL,
    )
    p = _make_tracked(callbacks=[cb], context="watch for failures")
    cog.bot._process_manager.list_processes.return_value = [p]
    await cog.process_list.callback(cog, interaction)
    call_kwargs = interaction.response.send_message.call_args[1]
    embeds = call_kwargs["embeds"]
    embed = embeds[0]
    # Find hooks field
    hooks_field = next((f for f in embed.fields if f.name == "Hooks"), None)
    assert hooks_field is not None
    assert "notify_channel" in hooks_field.value
    assert "on_exit" in hooks_field.value
    # Context in its own field
    ctx_field = next((f for f in embed.fields if f.name == "Context"), None)
    assert ctx_field is not None
    assert "watch for failures" in ctx_field.value


@pytest.mark.asyncio
async def test_process_list_no_manager(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """Commands handle missing process manager gracefully."""
    cog.bot._process_manager = None
    await cog.process_list.callback(cog, interaction)
    assert "not available" in str(interaction.response.send_message.call_args).lower()


# ---------------------------------------------------------------------------
# process_list — embed details
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_list_discord_timestamp(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """Started field uses Discord relative timestamp format."""
    p = _make_tracked(started_at="2026-02-14T12:00:00+00:00")
    cog.bot._process_manager.list_processes.return_value = [p]
    await cog.process_list.callback(cog, interaction)
    embeds = interaction.response.send_message.call_args[1]["embeds"]
    started_field = next(f for f in embeds[0].fields if f.name == "Started")
    assert started_field.value.startswith("<t:")
    assert started_field.value.endswith(":R>")


@pytest.mark.asyncio
async def test_process_list_working_directory(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """Each embed has a Working Directory field."""
    p = _make_tracked(working_directory="/home/agent/workspace")
    cog.bot._process_manager.list_processes.return_value = [p]
    await cog.process_list.callback(cog, interaction)
    embeds = interaction.response.send_message.call_args[1]["embeds"]
    wd_field = next(f for f in embeds[0].fields if f.name == "Working Directory")
    assert "/home/agent/workspace" in wd_field.value


@pytest.mark.asyncio
async def test_process_list_exit_code_field(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """Exit code appears as a dedicated field for exited processes."""
    p = _make_tracked(status=ProcessStatus.EXITED, exit_code=1)
    cog.bot._process_manager.list_processes.return_value = [p]
    await cog.process_list.callback(cog, interaction)
    embeds = interaction.response.send_message.call_args[1]["embeds"]
    exit_field = next(f for f in embeds[0].fields if f.name == "Exit Code")
    assert exit_field.value == "1"


@pytest.mark.asyncio
async def test_process_list_context_not_truncated_at_100(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """Context field allows up to 1000 chars, not the old 100-char limit."""
    long_context = "A" * 500
    p = _make_tracked(context=long_context)
    cog.bot._process_manager.list_processes.return_value = [p]
    await cog.process_list.callback(cog, interaction)
    embeds = interaction.response.send_message.call_args[1]["embeds"]
    ctx_field = next(f for f in embeds[0].fields if f.name == "Context")
    # Full 500 chars should be present (not truncated at 100)
    assert ctx_field.value == long_context


@pytest.mark.asyncio
async def test_process_list_footer_agent_name(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """Each embed has footer with agent name."""
    p = _make_tracked(agent_name="test-agent")
    cog.bot._process_manager.list_processes.return_value = [p]
    await cog.process_list.callback(cog, interaction)
    embeds = interaction.response.send_message.call_args[1]["embeds"]
    assert embeds[0].footer.text == "Agent: test-agent"


# ---------------------------------------------------------------------------
# Hook formatting
# ---------------------------------------------------------------------------


def test_format_hook_fire_count() -> None:
    """Active hook shows fire count as N/M."""
    cb = ProcessCallback(
        trigger=HookTrigger(type=TriggerType.ON_EXIT, exit_filter=ExitFilter.FAILURE),
        action=CallbackAction.NOTIFY_CHANNEL,
        max_fires=3,
        fire_count=1,
    )
    result = _format_hook(cb)
    assert "1/3" in result
    assert "on_exit" in result
    assert "notify_channel" in result


def test_format_hook_output_delay() -> None:
    """Hook with output_delay_seconds shows delay info."""
    cb = ProcessCallback(
        trigger=HookTrigger(type=TriggerType.ON_OUTPUT_MATCH, pattern="ERROR"),
        action=CallbackAction.SPAWN_BRANCH,
        output_delay_seconds=2.0,
    )
    result = _format_hook(cb)
    assert "delay 2.0s" in result
    assert "ERROR" in result


def test_format_hook_exhausted_strikethrough() -> None:
    """Exhausted hook uses strikethrough and (exhausted) label."""
    cb = ProcessCallback(
        trigger=HookTrigger(type=TriggerType.ON_TIMEOUT, timeout_seconds=600),
        action=CallbackAction.STOP_PROCESS,
        max_fires=1,
        fire_count=1,  # exhausted
    )
    result = _format_hook(cb)
    assert result.startswith("~~")
    assert "(exhausted)" in result
    assert "on_timeout" in result
    assert "600" in result


def test_format_hook_pattern_match() -> None:
    """Output match hook shows the pattern."""
    cb = ProcessCallback(
        trigger=HookTrigger(type=TriggerType.ON_OUTPUT_MATCH, pattern="FATAL"),
        action=CallbackAction.INJECT_CONTEXT,
    )
    result = _format_hook(cb)
    assert "`FATAL`" in result


# ---------------------------------------------------------------------------
# Overflow embed (>10 processes)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_list_overflow(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """More than 10 processes: 9 full embeds + 1 overflow summary."""
    processes = [_make_tracked(pid=1000 + i) for i in range(12)]
    cog.bot._process_manager.list_processes.return_value = processes
    await cog.process_list.callback(cog, interaction)
    embeds = interaction.response.send_message.call_args[1]["embeds"]
    assert len(embeds) == 10
    # First 9 are full process embeds
    for i in range(9):
        assert f"PID {1000 + i}" in embeds[i].title
    # 10th is overflow summary
    overflow = embeds[9]
    assert "3 more" in overflow.title
    # Overflow lists remaining PIDs
    assert "1009" in overflow.description
    assert "1010" in overflow.description
    assert "1011" in overflow.description


@pytest.mark.asyncio
async def test_process_list_exactly_10(
    cog: ProcessCog, interaction: MagicMock
) -> None:
    """Exactly 10 processes: all get full embeds, no overflow."""
    processes = [_make_tracked(pid=2000 + i) for i in range(10)]
    cog.bot._process_manager.list_processes.return_value = processes
    await cog.process_list.callback(cog, interaction)
    embeds = interaction.response.send_message.call_args[1]["embeds"]
    assert len(embeds) == 10
    # All should be full embeds (check title format)
    for i in range(10):
        assert f"PID {2000 + i}" in embeds[i].title


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_discord_timestamp_valid_iso() -> None:
    """Converts valid ISO timestamp to Discord format."""
    result = _discord_timestamp("2026-02-14T12:00:00+00:00")
    assert result.startswith("<t:")
    assert result.endswith(":R>")
    # Extract the unix timestamp
    ts = int(result[3:-3])
    assert ts > 0


def test_discord_timestamp_naive() -> None:
    """Handles naive (no timezone) ISO timestamp."""
    result = _discord_timestamp("2026-02-14T12:00:00")
    assert result.startswith("<t:")


def test_discord_timestamp_invalid() -> None:
    """Falls back to raw string for invalid timestamps."""
    result = _discord_timestamp("not-a-timestamp-at-all")
    assert result == "not-a-timestamp-at-"  # truncated to 19 chars


def test_safe_truncate_short() -> None:
    """Short text passes through unchanged."""
    assert _safe_truncate("hello", 10) == "hello"


def test_safe_truncate_long() -> None:
    """Long text is truncated with ellipsis."""
    result = _safe_truncate("a" * 20, 10)
    assert len(result) == 10
    assert result.endswith("...")


def test_build_process_embed_color_running() -> None:
    """Running process gets green color."""
    p = _make_tracked(status=ProcessStatus.RUNNING)
    embed = _build_process_embed(p)
    assert embed.color == discord.Color.green()


def test_build_process_embed_color_error_exit() -> None:
    """Exited process with non-zero exit code gets red color."""
    p = _make_tracked(status=ProcessStatus.EXITED, exit_code=1)
    embed = _build_process_embed(p)
    assert embed.color == discord.Color.red()


def test_build_process_embed_color_clean_exit() -> None:
    """Exited process with exit code 0 gets grey color."""
    p = _make_tracked(status=ProcessStatus.EXITED, exit_code=0)
    embed = _build_process_embed(p)
    assert embed.color == discord.Color.greyple()


def test_build_overflow_embed() -> None:
    """Overflow embed lists processes compactly."""
    processes = [_make_tracked(pid=100 + i) for i in range(3)]
    embed = _build_overflow_embed(processes)
    assert "3 more" in embed.title
    assert "PID 100" in embed.description
    assert "PID 101" in embed.description
    assert "PID 102" in embed.description


# ---------------------------------------------------------------------------
# process_kill and process_logs — unchanged behavior
# ---------------------------------------------------------------------------


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
