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
    GlobalEditRateLimiter,
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

    def test_embed_title_contains_agent_and_branch(self) -> None:
        snap = self._make_snapshot()
        embed = build_status_embed(snap)
        assert "test-agent" in embed.title
        assert "#1" in embed.title

    def test_embed_color_blue_for_processing(self) -> None:
        snap = self._make_snapshot(status="processing")
        embed = build_status_embed(snap)
        assert embed.colour == discord.Colour.blue()

    def test_embed_color_blue_for_waiting(self) -> None:
        snap = self._make_snapshot(status="waiting")
        embed = build_status_embed(snap)
        assert embed.colour == discord.Colour.blue()

    def test_embed_color_blue_for_completed(self) -> None:
        snap = self._make_snapshot(status="completed")
        embed = build_status_embed(snap)
        assert embed.colour == discord.Colour.blue()

    def test_embed_color_red_for_error(self) -> None:
        snap = self._make_snapshot(status="error")
        embed = build_status_embed(snap)
        assert embed.colour == discord.Colour.red()

    def test_embed_color_red_for_cancelled(self) -> None:
        snap = self._make_snapshot(status="cancelled")
        embed = build_status_embed(snap)
        assert embed.colour == discord.Colour.red()

    def test_description_contains_status(self) -> None:
        snap = self._make_snapshot(status="processing")
        embed = build_status_embed(snap)
        assert "Processing" in embed.description

    def test_description_contains_step_info(self) -> None:
        snap = self._make_snapshot(current_step="Running bash", step_number=3)
        embed = build_status_embed(snap)
        assert "3" in embed.description
        assert "bash" in embed.description

    def test_description_contains_tokens(self) -> None:
        snap = self._make_snapshot(
            token_usage=Usage(input_tokens=1247, output_tokens=892),
        )
        embed = build_status_embed(snap)
        assert "1,247" in embed.description
        assert "892" in embed.description

    def test_description_contains_elapsed(self) -> None:
        snap = self._make_snapshot(elapsed_ms=5200)
        embed = build_status_embed(snap)
        assert "5.2s" in embed.description

    def test_description_contains_tools(self) -> None:
        snap = self._make_snapshot(tool_calls_made=3, tools_used=["bash", "create_file"])
        embed = build_status_embed(snap)
        assert "3 tools" in embed.description

    def test_description_contains_llm_calls(self) -> None:
        snap = self._make_snapshot(llm_iterations=4)
        embed = build_status_embed(snap)
        assert "4 calls" in embed.description

    def test_completed_uses_steps_language(self) -> None:
        snap = self._make_snapshot(status="completed", step_number=5)
        embed = build_status_embed(snap)
        assert "5 steps" in embed.description

    def test_error_shows_error_message(self) -> None:
        snap = self._make_snapshot(status="error", error_message="API timeout")
        embed = build_status_embed(snap)
        assert "API timeout" in embed.description

    def test_no_fields_used(self) -> None:
        """Embed should use description only, no fields."""
        snap = self._make_snapshot()
        embed = build_status_embed(snap)
        assert len(embed.fields) == 0

    def test_single_call_uses_singular(self) -> None:
        snap = self._make_snapshot(llm_iterations=1)
        embed = build_status_embed(snap)
        assert "1 call" in embed.description
        assert "1 calls" not in embed.description

    # ── Finalized with response_content ──────────────────────────────

    def test_finalized_embed_shows_response_as_description(self) -> None:
        snap = self._make_snapshot(
            status="completed",
            response_content="Here is the answer to your question.",
            step_number=3,
            token_usage=Usage(input_tokens=100, output_tokens=50),
            elapsed_ms=2500,
        )
        embed = build_status_embed(snap)
        assert embed.description == "Here is the answer to your question."
        assert embed.title == "test-agent"  # Just agent name, no branch

    def test_finalized_embed_has_footer_with_metrics(self) -> None:
        snap = self._make_snapshot(
            status="completed",
            response_content="Response text",
            step_number=5,
            token_usage=Usage(input_tokens=1000, output_tokens=200),
            elapsed_ms=3200,
        )
        embed = build_status_embed(snap)
        footer = embed.footer.text
        assert "branch #1" in footer
        assert "5 steps" in footer
        assert "1,000 in / 200 out" in footer
        assert "3.2s" in footer

    def test_finalized_embed_truncates_long_response(self) -> None:
        long_content = "x" * 5000
        snap = self._make_snapshot(
            status="completed",
            response_content=long_content,
        )
        embed = build_status_embed(snap)
        assert len(embed.description) <= 4001  # 4000 + "…"
        assert embed.description.endswith("…")

    def test_finalized_embed_short_response_not_truncated(self) -> None:
        snap = self._make_snapshot(
            status="completed",
            response_content="Short answer",
        )
        embed = build_status_embed(snap)
        assert embed.description == "Short answer"

    def test_finalized_embed_uses_blue_color(self) -> None:
        snap = self._make_snapshot(
            status="completed",
            response_content="Done!",
        )
        embed = build_status_embed(snap)
        assert embed.colour == discord.Colour.blue()

    def test_finalized_embed_error_appended(self) -> None:
        snap = self._make_snapshot(
            status="error",
            response_content="Partial output",
            error_message="Something failed",
        )
        embed = build_status_embed(snap)
        assert "Partial output" in embed.description
        assert "Something failed" in embed.description


# ---------------------------------------------------------------------------
# GlobalEditRateLimiter
# ---------------------------------------------------------------------------


class FakeClock:
    """Deterministic clock for testing throttle behavior."""

    def __init__(self, start: float = 0.0) -> None:
        self._time = start

    def __call__(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


class TestGlobalEditRateLimiter:
    def test_can_edit_initially(self) -> None:
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        assert rl.can_edit_now() is True

    def test_cannot_edit_immediately_after_record(self) -> None:
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        rl.record_edit()
        clock.advance(0.5)
        assert rl.can_edit_now() is False

    def test_can_edit_after_interval(self) -> None:
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        rl.record_edit()
        clock.advance(1.2)
        assert rl.can_edit_now() is True

    def test_time_until_next_allowed_zero_initially(self) -> None:
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        assert rl.time_until_next_allowed() == 0.0

    def test_time_until_next_allowed_after_record(self) -> None:
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        rl.record_edit()
        clock.advance(0.3)
        remaining = rl.time_until_next_allowed()
        assert 0.7 < remaining < 0.9  # ~0.8

    def test_time_until_next_allowed_zero_after_interval(self) -> None:
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        rl.record_edit()
        clock.advance(2.0)
        assert rl.time_until_next_allowed() == 0.0

    def test_shared_across_two_views(self) -> None:
        """A rate limiter shared between two views: second is throttled by first's edit."""
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        # First "view" records an edit
        rl.record_edit()
        clock.advance(0.5)
        # Second "view" should be blocked
        assert rl.can_edit_now() is False
        clock.advance(0.7)  # Now 1.2s total
        assert rl.can_edit_now() is True


# ---------------------------------------------------------------------------
# LiveStatusView
# ---------------------------------------------------------------------------


def _make_mock_channel() -> MagicMock:
    channel = MagicMock(spec=discord.TextChannel)
    mock_message = MagicMock(spec=discord.Message)
    mock_message.edit = AsyncMock()
    mock_message.id = 12345
    channel.send = AsyncMock(return_value=mock_message)
    return channel


class TestLiveStatusView:
    @pytest.mark.asyncio
    async def test_start_sends_initial_embed(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _rate_limiter=rl,
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
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _rate_limiter=rl,
            _clock=clock,
        )
        await view.start()
        clock.advance(2.0)  # Past throttle interval
        await view.update(current_step="Running bash", step_number=1)
        await asyncio.sleep(0)
        msg = channel.send.return_value
        msg.edit.assert_called()

    @pytest.mark.asyncio
    async def test_throttles_rapid_edits(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.5, _clock=clock)
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _rate_limiter=rl,
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
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _rate_limiter=rl,
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
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _rate_limiter=rl,
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
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _rate_limiter=rl,
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
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        view1 = LiveStatusView(
            channel=channel,
            agent_name="agent",
            thread_id=1,
            get_active_count=lambda: 2,
            _rate_limiter=rl,
            _clock=clock,
        )
        view2 = LiveStatusView(
            channel=channel,
            agent_name="agent",
            thread_id=2,
            get_active_count=lambda: 2,
            _rate_limiter=rl,
            _clock=clock,
        )
        await view1.start()
        await view2.start()
        assert channel.send.call_count == 2

    @pytest.mark.asyncio
    async def test_batches_rapid_events(self) -> None:
        """Multiple updates within throttle window -> latest values in single edit."""
        channel = _make_mock_channel()
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.5, _clock=clock)
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _rate_limiter=rl,
            _clock=clock,
        )
        await view.start()
        clock.advance(2.0)  # Past throttle -> first update edits immediately
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
        assert embed.colour == discord.Colour.blue()

    # ── Reference parameter ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reference_passed_to_channel_send(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        ref_message = MagicMock(spec=discord.Message)
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            reference=ref_message,
            _rate_limiter=rl,
            _clock=clock,
        )
        await view.start()
        call_kwargs = channel.send.call_args.kwargs
        assert call_kwargs["reference"] is ref_message

    @pytest.mark.asyncio
    async def test_no_reference_omits_param(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _rate_limiter=rl,
            _clock=clock,
        )
        await view.start()
        call_kwargs = channel.send.call_args.kwargs
        assert "reference" not in call_kwargs

    # ── message property ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_message_property_none_before_start(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _rate_limiter=rl,
            _clock=clock,
        )
        assert view.message is None

    @pytest.mark.asyncio
    async def test_message_property_returns_sent_message(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _rate_limiter=rl,
            _clock=clock,
        )
        await view.start()
        assert view.message is channel.send.return_value

    # ── finalize with response_content ───────────────────────────────

    @pytest.mark.asyncio
    async def test_finalize_with_response_content(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _rate_limiter=rl,
            _clock=clock,
        )
        await view.start()
        clock.advance(5.0)
        await view.finalize("completed", response_content="Here is the answer.")
        msg = channel.send.return_value
        final_call = msg.edit.call_args
        embed = final_call.kwargs["embed"]
        # Description should be the response content
        assert embed.description == "Here is the answer."
        # Title should be just agent name
        assert embed.title == "test-agent"
        # Footer should contain metrics
        assert "branch #1" in embed.footer.text

    @pytest.mark.asyncio
    async def test_finalize_response_content_truncation(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _rate_limiter=rl,
            _clock=clock,
        )
        await view.start()
        long_response = "a" * 5000
        await view.finalize("completed", response_content=long_response)
        msg = channel.send.return_value
        final_call = msg.edit.call_args
        embed = final_call.kwargs["embed"]
        assert len(embed.description) <= 4001
        assert embed.description.endswith("…")

    # ── elapsed_ms auto-tracking ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_elapsed_ms_auto_tracked(self) -> None:
        channel = _make_mock_channel()
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        view = LiveStatusView(
            channel=channel,
            agent_name="test-agent",
            thread_id=1,
            get_active_count=lambda: 1,
            _rate_limiter=rl,
            _clock=clock,
        )
        await view.start()
        clock.advance(3.5)  # 3500ms
        await view.finalize("completed", response_content="Done")
        msg = channel.send.return_value
        final_call = msg.edit.call_args
        embed = final_call.kwargs["embed"]
        # Footer should contain 3.5s
        assert "3.5s" in embed.footer.text

    # ── Global rate limiter shared across views ──────────────────────

    @pytest.mark.asyncio
    async def test_global_rate_limiter_shared_between_views(self) -> None:
        """Second view's update is throttled by first view's edit."""
        channel = _make_mock_channel()
        clock = FakeClock()
        rl = GlobalEditRateLimiter(min_interval=1.1, _clock=clock)
        view1 = LiveStatusView(
            channel=channel,
            agent_name="agent",
            thread_id=1,
            get_active_count=lambda: 2,
            _rate_limiter=rl,
            _clock=clock,
        )
        view2 = LiveStatusView(
            channel=channel,
            agent_name="agent",
            thread_id=2,
            get_active_count=lambda: 2,
            _rate_limiter=rl,
            _clock=clock,
        )
        await view1.start()
        clock.advance(1.2)  # Past interval
        await view1.update(current_step="Step 1")
        msg = channel.send.return_value
        assert msg.edit.call_count == 1

        # view2 update immediately after view1 edit — should be throttled
        clock.advance(0.1)
        await view2.update(current_step="Step 1")
        await asyncio.sleep(0)
        # edit was from view1 only; view2's is deferred
        assert msg.edit.call_count == 1


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
        # At most one more call (the deferred one hasn't fired yet)
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
