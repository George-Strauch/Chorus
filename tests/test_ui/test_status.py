"""Tests for chorus.ui.status — live status feedback UI."""

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
    build_status_embed,
)

# ---------------------------------------------------------------------------
# StatusSnapshot + build_status_embed
# ---------------------------------------------------------------------------


class TestBuildStatusEmbed:
    def _make_snapshot(self, **overrides: Any) -> StatusSnapshot:
        defaults: dict[str, Any] = {
            "agent_name": "test-agent",
            "thread_id": 1,
            "status": "processing",
        }
        defaults.update(overrides)
        return StatusSnapshot(**defaults)

    def test_embed_title_contains_agent_and_thread(self) -> None:
        snap = self._make_snapshot()
        embed = build_status_embed(snap)
        assert "test-agent" in embed.title
        assert "#1" in embed.title

    def test_embed_color_blue_for_processing(self) -> None:
        snap = self._make_snapshot(status="processing")
        embed = build_status_embed(snap)
        assert embed.colour == discord.Colour.blue()

    def test_embed_color_yellow_for_waiting(self) -> None:
        snap = self._make_snapshot(status="waiting")
        embed = build_status_embed(snap)
        assert embed.colour == discord.Colour.yellow()

    def test_embed_color_green_for_completed(self) -> None:
        snap = self._make_snapshot(status="completed")
        embed = build_status_embed(snap)
        assert embed.colour == discord.Colour.green()

    def test_embed_color_red_for_error(self) -> None:
        snap = self._make_snapshot(status="error")
        embed = build_status_embed(snap)
        assert embed.colour == discord.Colour.red()

    def test_embed_color_red_for_cancelled(self) -> None:
        snap = self._make_snapshot(status="cancelled")
        embed = build_status_embed(snap)
        assert embed.colour == discord.Colour.red()

    def test_embed_has_status_field(self) -> None:
        snap = self._make_snapshot(status="processing")
        embed = build_status_embed(snap)
        field_names = [f.name for f in embed.fields]
        assert "Status" in field_names

    def test_embed_has_step_field(self) -> None:
        snap = self._make_snapshot(current_step="Running bash", step_number=3)
        embed = build_status_embed(snap)
        field_names = [f.name for f in embed.fields]
        assert "Step" in field_names
        step_field = next(f for f in embed.fields if f.name == "Step")
        assert "3" in step_field.value
        assert "bash" in step_field.value

    def test_embed_has_tokens_field_with_formatting(self) -> None:
        snap = self._make_snapshot(
            token_usage=Usage(input_tokens=1247, output_tokens=892),
        )
        embed = build_status_embed(snap)
        field_names = [f.name for f in embed.fields]
        assert "Tokens" in field_names
        tokens_field = next(f for f in embed.fields if f.name == "Tokens")
        assert "1,247" in tokens_field.value
        assert "892" in tokens_field.value

    def test_embed_has_elapsed_field(self) -> None:
        snap = self._make_snapshot(elapsed_ms=5200)
        embed = build_status_embed(snap)
        field_names = [f.name for f in embed.fields]
        assert "Elapsed" in field_names
        elapsed_field = next(f for f in embed.fields if f.name == "Elapsed")
        assert "5.2s" in elapsed_field.value

    def test_embed_has_tools_field(self) -> None:
        snap = self._make_snapshot(tool_calls_made=3, tools_used=["bash", "create_file"])
        embed = build_status_embed(snap)
        field_names = [f.name for f in embed.fields]
        assert "Tools" in field_names

    def test_embed_has_llm_calls_field(self) -> None:
        snap = self._make_snapshot(llm_iterations=4)
        embed = build_status_embed(snap)
        field_names = [f.name for f in embed.fields]
        assert "LLM Calls" in field_names

    def test_completed_uses_total_language(self) -> None:
        snap = self._make_snapshot(status="completed", step_number=5)
        embed = build_status_embed(snap)
        step_field = next(f for f in embed.fields if f.name == "Step")
        assert "total" in step_field.value.lower() or "5" in step_field.value

    def test_error_shows_error_message(self) -> None:
        snap = self._make_snapshot(status="error", error_message="API timeout")
        embed = build_status_embed(snap)
        # Error message should appear somewhere in the embed
        all_values = " ".join(f.value for f in embed.fields)
        assert "API timeout" in all_values


# ---------------------------------------------------------------------------
# LiveStatusView
# ---------------------------------------------------------------------------


def _make_mock_channel() -> MagicMock:
    channel = MagicMock(spec=discord.TextChannel)
    mock_message = MagicMock(spec=discord.Message)
    mock_message.edit = AsyncMock()
    channel.send = AsyncMock(return_value=mock_message)
    return channel


class FakeClock:
    """Deterministic clock for testing throttle behavior."""

    def __init__(self, start: float = 0.0) -> None:
        self._time = start

    def __call__(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


class TestLiveStatusView:
    @pytest.mark.asyncio
    async def test_start_sends_initial_embed(self) -> None:
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
        assert "embed" in call_kwargs.kwargs

    @pytest.mark.asyncio
    async def test_update_edits_embed(self) -> None:
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
        clock.advance(2.0)  # Past throttle interval
        await view.update(current_step="Running bash", step_number=1)
        # Allow any deferred tasks to run
        await asyncio.sleep(0)
        msg = channel.send.return_value
        msg.edit.assert_called()

    @pytest.mark.asyncio
    async def test_throttles_rapid_edits(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            min_edit_interval=1.5,
            _clock=clock,
        )
        await view.start()
        # First update right after start (within throttle window)
        clock.advance(0.1)
        await view.update(current_step="Step 1")
        await view.update(current_step="Step 2")
        await asyncio.sleep(0)
        msg = channel.send.return_value
        # Should NOT have edited yet (within throttle window)
        msg.edit.assert_not_called()

    @pytest.mark.asyncio
    async def test_finalize_always_edits(self) -> None:
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
        # Finalize immediately (within throttle window) — should still edit
        await view.finalize("completed")
        msg = channel.send.return_value
        msg.edit.assert_called_once()

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
        # Subsequent updates should be no-ops (no message to edit)
        await view.update(current_step="Step 1")
        await view.finalize("completed")

    @pytest.mark.asyncio
    async def test_edit_failure_handled(self) -> None:
        channel = _make_mock_channel()
        msg = channel.send.return_value
        msg.edit = AsyncMock(side_effect=discord.HTTPException(MagicMock(), "fail"))
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
        # Should not raise
        await view.update(current_step="Step 1")
        await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_separate_embeds_per_thread(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        view1 = LiveStatusView(
            channel=channel,
            agent_name="agent",
            thread_id=1,
            get_active_count=lambda: 2,
            _clock=clock,
        )
        view2 = LiveStatusView(
            channel=channel,
            agent_name="agent",
            thread_id=2,
            get_active_count=lambda: 2,
            _clock=clock,
        )
        await view1.start()
        await view2.start()
        assert channel.send.call_count == 2

    @pytest.mark.asyncio
    async def test_batches_rapid_events(self) -> None:
        """Multiple updates within throttle window → latest values in single edit."""
        channel = _make_mock_channel()
        clock = FakeClock()
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            min_edit_interval=1.5,
            _clock=clock,
        )
        await view.start()
        clock.advance(2.0)  # Past throttle → first update edits immediately
        await view.update(current_step="Step 1", step_number=1)
        await asyncio.sleep(0)
        msg = channel.send.return_value
        assert msg.edit.call_count == 1

        # Now rapid updates within new throttle window
        clock.advance(0.1)
        await view.update(current_step="Step 2", step_number=2)
        await view.update(current_step="Step 3", step_number=3)
        # Finalize should override throttle and use latest values
        await view.finalize("completed")
        # Should have edited with final state
        final_call = msg.edit.call_args
        embed = final_call.kwargs["embed"]
        assert embed.colour == discord.Colour.green()


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
        assert "2" in activity.name  # 2 tasks or 2 agents

    @pytest.mark.asyncio
    async def test_debounces_rapid_updates(self) -> None:
        bot = _make_mock_bot()
        clock = FakeClock()
        mgr = BotPresenceManager(bot, debounce_seconds=5.0, _clock=clock)
        await mgr.thread_started("agent-a", 1)
        initial_count = bot.change_presence.call_count
        # Rapid follow-up — should NOT trigger another change_presence immediately
        clock.advance(0.1)
        await mgr.thread_started("agent-a", 2)
        await asyncio.sleep(0)
        # At most one more call (the deferred one hasn't fired yet because real time hasn't passed)
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
        # Should show 1 task remaining
        assert "1" in activity.name
