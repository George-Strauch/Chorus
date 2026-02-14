"""Tests for chorus.ui.status — plain text status + response formatting."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from chorus.llm.providers import Usage
from chorus.ui.status import (
    BotPresenceManager,
    LiveStatusView,
    StatusSnapshot,
    format_response_footer,
    format_status_line,
)

# ---------------------------------------------------------------------------
# format_response_footer
# ---------------------------------------------------------------------------


class TestFormatResponseFooter:
    def _make_snapshot(self, **overrides: Any) -> StatusSnapshot:
        defaults: dict[str, Any] = {
            "agent_name": "test-agent",
            "thread_id": 1,
            "status": "completed",
        }
        defaults.update(overrides)
        return StatusSnapshot(**defaults)

    def test_footer_contains_branch_id(self) -> None:
        snap = self._make_snapshot(thread_id=3)
        footer = format_response_footer(snap)
        assert "branch #3" in footer

    def test_footer_contains_step_count(self) -> None:
        snap = self._make_snapshot(step_number=5)
        footer = format_response_footer(snap)
        assert "5 steps" in footer

    def test_footer_contains_token_counts(self) -> None:
        snap = self._make_snapshot(
            token_usage=Usage(input_tokens=1247, output_tokens=892),
        )
        footer = format_response_footer(snap)
        assert "1,247 in" in footer
        assert "892 out" in footer

    def test_footer_contains_elapsed(self) -> None:
        snap = self._make_snapshot(elapsed_ms=5200)
        footer = format_response_footer(snap)
        assert "5.2s" in footer

    def test_footer_is_italic(self) -> None:
        snap = self._make_snapshot()
        footer = format_response_footer(snap)
        assert footer.startswith("*")
        assert footer.endswith("*")


# ---------------------------------------------------------------------------
# LiveStatusView
# ---------------------------------------------------------------------------


class FakeClock:
    """Deterministic clock for testing."""

    def __init__(self, start: float = 0.0) -> None:
        self._time = start

    def __call__(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


def _make_mock_channel() -> MagicMock:
    channel = MagicMock(spec=discord.TextChannel)
    mock_message = MagicMock(spec=discord.Message)
    mock_message.edit = AsyncMock()
    mock_message.id = 12345
    channel.send = AsyncMock(return_value=mock_message)
    return channel


class TestLiveStatusView:
    @pytest.mark.asyncio
    async def test_start_sends_thinking_message(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        await view.start()
        channel.send.assert_called_once()
        call_kwargs = channel.send.call_args
        assert call_kwargs.kwargs["content"] == "Thinking..."
        view._stop_ticker()

    @pytest.mark.asyncio
    async def test_start_sends_plain_text_not_embed(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        await view.start()
        call_kwargs = channel.send.call_args.kwargs
        assert "embed" not in call_kwargs
        view._stop_ticker()

    @pytest.mark.asyncio
    async def test_update_stores_metrics_for_ticker(self) -> None:
        """update() stores metrics but doesn't directly edit the message."""
        channel = _make_mock_channel()
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        await view.start()
        clock.advance(2.0)
        await view.update(current_step="Running bash", step_number=1)
        # update() itself doesn't edit — the ticker does on its own schedule
        assert view._snapshot.current_step == "Running bash"
        assert view._snapshot.step_number == 1
        view._stop_ticker()

    @pytest.mark.asyncio
    async def test_finalize_under_15s_edits_in_place(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        await view.start()
        clock.advance(5.0)  # Under 15s
        await view.finalize("completed", response_content="Here is the answer.")
        msg = channel.send.return_value
        msg.edit.assert_called_once()
        edit_kwargs = msg.edit.call_args.kwargs
        assert "Here is the answer." in edit_kwargs["content"]
        # Should NOT have sent a second message
        assert channel.send.call_count == 1

    @pytest.mark.asyncio
    async def test_finalize_over_15s_sends_new_message(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        await view.start()
        clock.advance(20.0)  # Over 15s
        await view.finalize("completed", response_content="Here is the answer.")
        msg = channel.send.return_value
        msg.edit.assert_not_called()
        # Should have sent a second message
        assert channel.send.call_count == 2
        second_call = channel.send.call_args
        assert "Here is the answer." in second_call.kwargs["content"]

    @pytest.mark.asyncio
    async def test_finalize_response_contains_footer(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        await view.start()
        await view.update(step_number=3, token_usage=Usage(100, 50))
        clock.advance(5.0)
        await view.finalize("completed", response_content="Done!")
        msg = channel.send.return_value
        edit_kwargs = msg.edit.call_args.kwargs
        content = edit_kwargs["content"]
        assert "branch #1" in content
        assert "3 steps" in content
        assert "5.0s" in content

    @pytest.mark.asyncio
    async def test_finalize_error_shows_error_message(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        await view.start()
        clock.advance(2.0)
        await view.finalize("error", error="API timeout")
        msg = channel.send.return_value
        edit_kwargs = msg.edit.call_args.kwargs
        assert "API timeout" in edit_kwargs["content"]

    @pytest.mark.asyncio
    async def test_send_failure_handled(self) -> None:
        channel = _make_mock_channel()
        channel.send = AsyncMock(side_effect=discord.HTTPException(MagicMock(), "fail"))
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        # Should not raise
        await view.start()
        await view.update(current_step="Step 1")
        await view.finalize("completed")

    @pytest.mark.asyncio
    async def test_reference_passed_to_channel_send(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        ref_message = MagicMock(spec=discord.Message)
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            reference=ref_message,
            _clock=clock,
        )
        await view.start()
        call_kwargs = channel.send.call_args.kwargs
        assert call_kwargs["reference"] is ref_message
        view._stop_ticker()

    @pytest.mark.asyncio
    async def test_no_reference_omits_param(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        await view.start()
        call_kwargs = channel.send.call_args.kwargs
        assert "reference" not in call_kwargs
        view._stop_ticker()

    @pytest.mark.asyncio
    async def test_message_property_none_before_start(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        assert view.message is None

    @pytest.mark.asyncio
    async def test_message_property_returns_sent_message(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        await view.start()
        assert view.message is channel.send.return_value
        view._stop_ticker()

    @pytest.mark.asyncio
    async def test_elapsed_ms_auto_tracked(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        await view.start()
        clock.advance(3.5)  # 3500ms
        await view.finalize("completed", response_content="Done")
        msg = channel.send.return_value
        edit_kwargs = msg.edit.call_args.kwargs
        assert "3.5s" in edit_kwargs["content"]

    @pytest.mark.asyncio
    async def test_no_response_content_shows_placeholder(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        await view.start()
        clock.advance(2.0)
        await view.finalize("completed")
        msg = channel.send.return_value
        edit_kwargs = msg.edit.call_args.kwargs
        assert "*(no response)*" in edit_kwargs["content"]

    @pytest.mark.asyncio
    async def test_finalize_over_15s_updates_message_property(self) -> None:
        """When sending a new message (>15s), the message property should update."""
        channel = _make_mock_channel()
        new_msg = MagicMock(spec=discord.Message)
        new_msg.id = 99999
        # First call returns "thinking" msg, second returns new response msg
        original_msg = MagicMock(spec=discord.Message)
        original_msg.id = 12345
        original_msg.edit = AsyncMock()
        channel.send = AsyncMock(side_effect=[original_msg, new_msg])
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        await view.start()
        assert view.message is original_msg
        clock.advance(20.0)
        await view.finalize("completed", response_content="Response")
        assert view.message is new_msg


# ---------------------------------------------------------------------------
# format_status_line
# ---------------------------------------------------------------------------


class TestFormatStatusLine:
    def _make_snapshot(self, **overrides: Any) -> StatusSnapshot:
        defaults: dict[str, Any] = {
            "agent_name": "test-agent",
            "thread_id": 1,
            "status": "processing",
        }
        defaults.update(overrides)
        return StatusSnapshot(**defaults)

    def test_includes_current_step(self) -> None:
        snap = self._make_snapshot(current_step="Running bash")
        line = format_status_line(snap, 3.2)
        assert "Running bash" in line

    def test_includes_elapsed_time(self) -> None:
        snap = self._make_snapshot()
        line = format_status_line(snap, 5.7)
        assert "5.7s" in line

    def test_includes_token_counts(self) -> None:
        snap = self._make_snapshot(
            token_usage=Usage(input_tokens=1234, output_tokens=567),
        )
        line = format_status_line(snap, 1.0)
        assert "1,234 in" in line
        assert "567 out" in line

    def test_includes_tool_call_count(self) -> None:
        snap = self._make_snapshot(tool_calls_made=5)
        line = format_status_line(snap, 2.0)
        assert "5 calls" in line

    def test_singular_call(self) -> None:
        snap = self._make_snapshot(tool_calls_made=1)
        line = format_status_line(snap, 1.0)
        assert "1 call" in line
        assert "calls" not in line

    def test_no_calls_omits_count(self) -> None:
        snap = self._make_snapshot(tool_calls_made=0)
        line = format_status_line(snap, 1.0)
        assert "call" not in line

    def test_is_italic(self) -> None:
        snap = self._make_snapshot()
        line = format_status_line(snap, 0.0)
        assert line.startswith("*")
        assert line.endswith("*")

    def test_default_step_is_starting(self) -> None:
        snap = self._make_snapshot()
        line = format_status_line(snap, 0.0)
        assert "Starting" in line


# ---------------------------------------------------------------------------
# Ticker lifecycle
# ---------------------------------------------------------------------------


class TestTickerLifecycle:
    @pytest.mark.asyncio
    async def test_start_creates_ticker_task(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        await view.start()
        assert view._ticker_task is not None
        assert not view._ticker_task.done()
        view._stop_ticker()

    @pytest.mark.asyncio
    async def test_finalize_cancels_ticker(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        await view.start()
        ticker = view._ticker_task
        assert ticker is not None
        await view.finalize("completed", response_content="Done")
        # Let the event loop process the cancellation
        await asyncio.sleep(0)
        assert ticker.done()

    @pytest.mark.asyncio
    async def test_no_ticker_when_send_fails(self) -> None:
        channel = _make_mock_channel()
        channel.send = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(), "fail"),
        )
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _clock=clock,
        )
        await view.start()
        assert view._ticker_task is None


# ---------------------------------------------------------------------------
# BotPresenceManager
# ---------------------------------------------------------------------------


def _make_mock_bot() -> MagicMock:
    bot = MagicMock()
    bot.change_presence = AsyncMock()
    return bot


class TestBotPresenceManager:
    @pytest.mark.asyncio
    async def test_shows_processing_when_active(self) -> None:
        bot = _make_mock_bot()
        clock = FakeClock()
        mgr = BotPresenceManager(bot, debounce_seconds=0.0, _clock=clock)
        await mgr.thread_started("agent-a", 1)
        await asyncio.sleep(0)
        bot.change_presence.assert_called()
        call_kwargs = bot.change_presence.call_args
        activity = call_kwargs.kwargs.get("activity") or call_kwargs.args[0]
        assert "1" in activity.name and "processing" in activity.name.lower()

    @pytest.mark.asyncio
    async def test_shows_idle_when_none(self) -> None:
        bot = _make_mock_bot()
        clock = FakeClock()
        mgr = BotPresenceManager(bot, debounce_seconds=0.0, _clock=clock)
        await mgr.thread_started("agent-a", 1)
        await asyncio.sleep(0)
        bot.change_presence.reset_mock()
        await mgr.thread_completed("agent-a", 1)
        await asyncio.sleep(0)
        bot.change_presence.assert_called()
        call_kwargs = bot.change_presence.call_args
        activity = call_kwargs.kwargs.get("activity") or call_kwargs.args[0]
        assert "idle" in activity.name.lower()

    @pytest.mark.asyncio
    async def test_counts_multiple_agents(self) -> None:
        bot = _make_mock_bot()
        clock = FakeClock()
        mgr = BotPresenceManager(bot, debounce_seconds=0.0, _clock=clock)
        await mgr.thread_started("agent-a", 1)
        await mgr.thread_started("agent-b", 2)
        await asyncio.sleep(0)
        call_kwargs = bot.change_presence.call_args
        activity = call_kwargs.kwargs.get("activity") or call_kwargs.args[0]
        assert "2" in activity.name

    @pytest.mark.asyncio
    async def test_debounces_rapid_updates(self) -> None:
        bot = _make_mock_bot()
        clock = FakeClock()
        mgr = BotPresenceManager(bot, debounce_seconds=5.0, _clock=clock)
        await mgr.thread_started("agent-a", 1)
        initial_count = bot.change_presence.call_count
        clock.advance(0.1)
        await mgr.thread_started("agent-a", 2)
        await asyncio.sleep(0)
        assert bot.change_presence.call_count <= initial_count + 1

    @pytest.mark.asyncio
    async def test_handles_change_presence_failure(self) -> None:
        bot = _make_mock_bot()
        bot.change_presence = AsyncMock(side_effect=discord.HTTPException(MagicMock(), "fail"))
        clock = FakeClock()
        mgr = BotPresenceManager(bot, debounce_seconds=0.0, _clock=clock)
        # Should not raise
        await mgr.thread_started("agent-a", 1)
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_thread_start_triggers_update(self) -> None:
        bot = _make_mock_bot()
        clock = FakeClock()
        mgr = BotPresenceManager(bot, debounce_seconds=0.0, _clock=clock)
        bot.change_presence.assert_not_called()
        await mgr.thread_started("agent-a", 1)
        await asyncio.sleep(0)
        bot.change_presence.assert_called()

    @pytest.mark.asyncio
    async def test_thread_complete_decrements(self) -> None:
        bot = _make_mock_bot()
        clock = FakeClock()
        mgr = BotPresenceManager(bot, debounce_seconds=0.0, _clock=clock)
        await mgr.thread_started("agent-a", 1)
        await mgr.thread_started("agent-a", 2)
        await asyncio.sleep(0)
        bot.change_presence.reset_mock()
        await mgr.thread_completed("agent-a", 1)
        await asyncio.sleep(0)
        bot.change_presence.assert_called()
        call_kwargs = bot.change_presence.call_args
        activity = call_kwargs.kwargs.get("activity") or call_kwargs.args[0]
        assert "1" in activity.name
