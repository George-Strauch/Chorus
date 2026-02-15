"""Tests for HookDispatcher — trigger evaluation, action dispatch, safety."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chorus.process.hooks import HookDispatcher
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


def _make_tracked(
    pid: int = 100,
    callbacks: list[ProcessCallback] | None = None,
    **overrides: object,
) -> TrackedProcess:
    defaults: dict[str, object] = {
        "pid": pid,
        "command": "python test.py",
        "working_directory": "/tmp",
        "agent_name": "test-agent",
        "started_at": "2026-01-01T00:00:00",
        "process_type": ProcessType.BACKGROUND,
        "status": ProcessStatus.RUNNING,
        "callbacks": callbacks or [],
        "spawned_by_branch": 1,
    }
    defaults.update(overrides)
    return TrackedProcess(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def mock_pm() -> MagicMock:
    pm = MagicMock()
    pm.kill_process = AsyncMock()
    return pm


@pytest.fixture
def mock_spawner() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def dispatcher(mock_pm: MagicMock, mock_spawner: AsyncMock) -> HookDispatcher:
    return HookDispatcher(
        process_manager=mock_pm,
        branch_spawner=mock_spawner,
        default_output_delay=0.0,  # No delay for tests
        max_recursion_depth=3,
    )


# ---------------------------------------------------------------------------
# ON_EXIT triggers
# ---------------------------------------------------------------------------


class TestOnExitTrigger:
    @pytest.mark.asyncio
    async def test_on_exit_any_fires(
        self, dispatcher: HookDispatcher, mock_pm: MagicMock
    ) -> None:
        """ON_EXIT with ANY filter fires on any exit code."""
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_EXIT, exit_filter=ExitFilter.ANY),
            action=CallbackAction.STOP_PROCESS,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        await dispatcher._on_exit(100, 0)
        assert cb.fire_count == 1
        mock_pm.kill_process.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_on_exit_success_filter(
        self, dispatcher: HookDispatcher, mock_pm: MagicMock
    ) -> None:
        """ON_EXIT with SUCCESS filter only fires on exit code 0."""
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_EXIT, exit_filter=ExitFilter.SUCCESS),
            action=CallbackAction.STOP_PROCESS,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        # Non-zero exit — should NOT fire
        await dispatcher._on_exit(100, 1)
        assert cb.fire_count == 0

        # Zero exit — SHOULD fire
        await dispatcher._on_exit(100, 0)
        assert cb.fire_count == 1

    @pytest.mark.asyncio
    async def test_on_exit_failure_filter(
        self, dispatcher: HookDispatcher, mock_pm: MagicMock
    ) -> None:
        """ON_EXIT with FAILURE filter only fires on non-zero exit."""
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_EXIT, exit_filter=ExitFilter.FAILURE),
            action=CallbackAction.STOP_PROCESS,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        # Zero exit — should NOT fire
        await dispatcher._on_exit(100, 0)
        assert cb.fire_count == 0

        # Non-zero — SHOULD fire
        await dispatcher._on_exit(100, 1)
        assert cb.fire_count == 1

    @pytest.mark.asyncio
    async def test_on_exit_respects_max_fires(
        self, dispatcher: HookDispatcher, mock_pm: MagicMock
    ) -> None:
        """Exhausted callbacks don't fire again."""
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_EXIT),
            action=CallbackAction.STOP_PROCESS,
            max_fires=1,
            fire_count=1,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        await dispatcher._on_exit(100, 0)
        assert cb.fire_count == 1  # Not incremented
        mock_pm.kill_process.assert_not_awaited()


# ---------------------------------------------------------------------------
# ON_OUTPUT_MATCH triggers
# ---------------------------------------------------------------------------


class TestOnOutputMatchTrigger:
    @pytest.mark.asyncio
    async def test_output_match_fires(
        self, dispatcher: HookDispatcher, mock_pm: MagicMock
    ) -> None:
        """ON_OUTPUT_MATCH fires when line matches pattern."""
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_OUTPUT_MATCH, pattern=r"ERROR"),
            action=CallbackAction.STOP_PROCESS,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        await dispatcher._on_line(100, "stdout", "All good")
        assert cb.fire_count == 0

        await dispatcher._on_line(100, "stdout", "ERROR: something failed")
        assert cb.fire_count == 1

    @pytest.mark.asyncio
    async def test_output_match_no_match(
        self, dispatcher: HookDispatcher, mock_pm: MagicMock
    ) -> None:
        """No firing when pattern doesn't match."""
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_OUTPUT_MATCH, pattern=r"FATAL"),
            action=CallbackAction.STOP_PROCESS,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        await dispatcher._on_line(100, "stdout", "just a warning")
        assert cb.fire_count == 0

    @pytest.mark.asyncio
    async def test_output_match_with_delay(
        self, mock_pm: MagicMock
    ) -> None:
        """Output delay accumulates output before firing."""
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_OUTPUT_MATCH, pattern=r"ERROR"),
            action=CallbackAction.STOP_PROCESS,
            output_delay_seconds=0.1,
        )
        tracked = _make_tracked(callbacks=[cb])
        tracked.rolling_tail = deque(["line1", "line2"], maxlen=100)
        mock_pm.get_process.return_value = tracked

        dispatcher = HookDispatcher(
            process_manager=mock_pm,
            default_output_delay=0.1,
        )

        await dispatcher._on_line(100, "stdout", "ERROR: test")
        # Should not have fired yet (delay pending)
        assert cb.fire_count == 0

        # Wait for delay
        await asyncio.sleep(0.2)
        assert cb.fire_count == 1


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------


class TestActionDispatch:
    @pytest.mark.asyncio
    async def test_stop_process_action(
        self, dispatcher: HookDispatcher, mock_pm: MagicMock
    ) -> None:
        """STOP_PROCESS calls kill_process."""
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_EXIT),
            action=CallbackAction.STOP_PROCESS,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        await dispatcher._on_exit(100, 0)
        mock_pm.kill_process.assert_awaited_with(100)

    @pytest.mark.asyncio
    async def test_stop_branch_action(
        self, mock_pm: MagicMock
    ) -> None:
        """STOP_BRANCH calls the thread kill callback."""
        kill_cb = AsyncMock()
        dispatcher = HookDispatcher(
            process_manager=mock_pm,
            thread_kill_callback=kill_cb,
            default_output_delay=0.0,
        )
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_EXIT),
            action=CallbackAction.STOP_BRANCH,
        )
        tracked = _make_tracked(callbacks=[cb], spawned_by_branch=5)
        mock_pm.get_process.return_value = tracked

        await dispatcher._on_exit(100, 0)
        kill_cb.assert_awaited_once_with("test-agent", 5)

    @pytest.mark.asyncio
    async def test_inject_context_action(
        self, mock_pm: MagicMock
    ) -> None:
        """INJECT_CONTEXT calls the inject callback."""
        inject_cb = AsyncMock()
        dispatcher = HookDispatcher(
            process_manager=mock_pm,
            inject_callback=inject_cb,
            default_output_delay=0.0,
        )
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_EXIT),
            action=CallbackAction.INJECT_CONTEXT,
            context_message="Build complete",
        )
        tracked = _make_tracked(callbacks=[cb], spawned_by_branch=3)
        mock_pm.get_process.return_value = tracked

        await dispatcher._on_exit(100, 0)
        inject_cb.assert_awaited_once()
        call_args = inject_cb.call_args
        assert call_args[0][0] == "test-agent"
        assert call_args[0][1] == 3
        assert "Build complete" in call_args[0][2]

    @pytest.mark.asyncio
    async def test_spawn_branch_action(
        self, dispatcher: HookDispatcher, mock_pm: MagicMock, mock_spawner: AsyncMock
    ) -> None:
        """SPAWN_BRANCH calls the branch spawner."""
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_EXIT),
            action=CallbackAction.SPAWN_BRANCH,
            context_message="Process done",
        )
        tracked = _make_tracked(callbacks=[cb], hook_recursion_depth=0)
        mock_pm.get_process.return_value = tracked

        await dispatcher._on_exit(100, 0)
        mock_spawner.spawn_hook_branch.assert_awaited_once()
        call_kwargs = mock_spawner.spawn_hook_branch.call_args[1]
        assert call_kwargs["agent_name"] == "test-agent"
        assert call_kwargs["recursion_depth"] == 1


# ---------------------------------------------------------------------------
# Safety limits
# ---------------------------------------------------------------------------


class TestSafety:
    @pytest.mark.asyncio
    async def test_max_recursion_depth(
        self, dispatcher: HookDispatcher, mock_pm: MagicMock, mock_spawner: AsyncMock
    ) -> None:
        """SPAWN_BRANCH blocked when recursion depth exceeded."""
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_EXIT),
            action=CallbackAction.SPAWN_BRANCH,
        )
        tracked = _make_tracked(callbacks=[cb], hook_recursion_depth=3)
        mock_pm.get_process.return_value = tracked

        await dispatcher._on_exit(100, 0)
        mock_spawner.spawn_hook_branch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_max_fires_respected(
        self, dispatcher: HookDispatcher, mock_pm: MagicMock
    ) -> None:
        """Callback only fires up to max_fires times."""
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_OUTPUT_MATCH, pattern=r"hit"),
            action=CallbackAction.STOP_PROCESS,
            max_fires=2,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        await dispatcher._on_line(100, "stdout", "hit 1")
        await dispatcher._on_line(100, "stdout", "hit 2")
        await dispatcher._on_line(100, "stdout", "hit 3")  # Should not fire

        assert cb.fire_count == 2
        assert mock_pm.kill_process.await_count == 2

    @pytest.mark.asyncio
    async def test_max_fires_zero_unlimited(
        self, dispatcher: HookDispatcher, mock_pm: MagicMock
    ) -> None:
        """max_fires=0 means unlimited — callback never becomes exhausted."""
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_OUTPUT_MATCH, pattern=r"hit"),
            action=CallbackAction.STOP_PROCESS,
            max_fires=0,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        for i in range(10):
            await dispatcher._on_line(100, "stdout", f"hit {i}")

        assert cb.fire_count == 10
        assert not cb.exhausted
        assert mock_pm.kill_process.await_count == 10

    @pytest.mark.asyncio
    async def test_no_spawner_configured(
        self, mock_pm: MagicMock
    ) -> None:
        """SPAWN_BRANCH action is a no-op when no spawner is configured."""
        dispatcher = HookDispatcher(
            process_manager=mock_pm,
            branch_spawner=None,
            default_output_delay=0.0,
        )
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_EXIT),
            action=CallbackAction.SPAWN_BRANCH,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        # Should not raise
        await dispatcher._on_exit(100, 0)
        assert cb.fire_count == 1

    @pytest.mark.asyncio
    async def test_wire_to_manager(
        self, dispatcher: HookDispatcher, mock_pm: MagicMock
    ) -> None:
        """wire_to_manager connects callbacks to the process manager."""
        dispatcher.wire_to_manager()
        mock_pm.set_callbacks.assert_called_once()
        call_kwargs = mock_pm.set_callbacks.call_args[1]
        assert call_kwargs["on_line"] is not None
        assert call_kwargs["on_exit"] is not None
        assert call_kwargs["on_spawn"] is not None


# ---------------------------------------------------------------------------
# ON_TIMEOUT triggers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# NOTIFY_CHANNEL action
# ---------------------------------------------------------------------------


class TestNotifyChannelAction:
    @pytest.mark.asyncio
    async def test_notify_callback_called(
        self, mock_pm: MagicMock
    ) -> None:
        """NOTIFY_CHANNEL calls the notify callback with correct args."""
        notify_cb = AsyncMock()
        dispatcher = HookDispatcher(
            process_manager=mock_pm,
            notify_callback=notify_cb,
            default_output_delay=0.0,
        )
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_EXIT),
            action=CallbackAction.NOTIFY_CHANNEL,
            context_message="Process done",
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        await dispatcher._on_exit(100, 0)
        notify_cb.assert_awaited_once()
        call_args = notify_cb.call_args[0]
        assert call_args[0] == "test-agent"
        assert "Process done" in call_args[1]
        assert call_args[2] is tracked

    @pytest.mark.asyncio
    async def test_notify_no_callback_is_noop(
        self, mock_pm: MagicMock
    ) -> None:
        """NOTIFY_CHANNEL is a no-op when no notify callback is configured."""
        dispatcher = HookDispatcher(
            process_manager=mock_pm,
            notify_callback=None,
            default_output_delay=0.0,
        )
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_EXIT),
            action=CallbackAction.NOTIFY_CHANNEL,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        # Should not raise
        await dispatcher._on_exit(100, 0)
        assert cb.fire_count == 1


# ---------------------------------------------------------------------------
# on_spawn
# ---------------------------------------------------------------------------


class TestOnSpawn:
    def test_on_spawn_starts_timeout_watchers(
        self, mock_pm: MagicMock
    ) -> None:
        """_on_spawn calls start_timeout_watchers."""
        dispatcher = HookDispatcher(
            process_manager=mock_pm,
            default_output_delay=0.0,
        )
        # No callbacks to watch, so just verify it doesn't error
        mock_pm.get_process.return_value = _make_tracked(callbacks=[])
        dispatcher._on_spawn(100)
        mock_pm.get_process.assert_called_with(100)

    @pytest.mark.asyncio
    async def test_on_spawn_starts_timeout_task(
        self, mock_pm: MagicMock
    ) -> None:
        """_on_spawn starts timeout watchers for timeout callbacks."""
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_TIMEOUT, timeout_seconds=0.1),
            action=CallbackAction.STOP_PROCESS,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        dispatcher = HookDispatcher(
            process_manager=mock_pm,
            default_output_delay=0.0,
        )
        dispatcher._on_spawn(100)

        # Wait for the timeout to fire
        await asyncio.sleep(0.2)
        assert cb.fire_count == 1
        mock_pm.kill_process.assert_awaited()


class TestStartNewTimeoutWatchers:
    @pytest.mark.asyncio
    async def test_start_new_timeout_watchers_only_new(
        self, mock_pm: MagicMock
    ) -> None:
        """start_new_timeout_watchers only starts watchers for the given callbacks."""
        # Existing callback (already has a watcher)
        existing_cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_TIMEOUT, timeout_seconds=10.0),
            action=CallbackAction.STOP_PROCESS,
        )
        # New callback to add
        new_cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_TIMEOUT, timeout_seconds=0.1),
            action=CallbackAction.STOP_PROCESS,
        )
        tracked = _make_tracked(callbacks=[existing_cb, new_cb])
        mock_pm.get_process.return_value = tracked

        dispatcher = HookDispatcher(
            process_manager=mock_pm,
            default_output_delay=0.0,
        )

        # Only pass the new callback
        dispatcher.start_new_timeout_watchers(100, [new_cb])

        # Wait for the short timeout to fire
        await asyncio.sleep(0.2)
        assert new_cb.fire_count == 1
        # Existing callback should NOT have fired (10s timeout, not started)
        assert existing_cb.fire_count == 0

    def test_start_new_timeout_watchers_ignores_non_timeout(
        self, mock_pm: MagicMock
    ) -> None:
        """start_new_timeout_watchers ignores non-timeout callbacks."""
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_EXIT),
            action=CallbackAction.NOTIFY_CHANNEL,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        dispatcher = HookDispatcher(
            process_manager=mock_pm,
            default_output_delay=0.0,
        )
        # Should not raise or create tasks
        dispatcher.start_new_timeout_watchers(100, [cb])
        # No timeout tasks should have been created
        assert 100 not in dispatcher._timeout_tasks


class TestOnTimeoutTrigger:
    @pytest.mark.asyncio
    async def test_timeout_fires(
        self, dispatcher: HookDispatcher, mock_pm: MagicMock
    ) -> None:
        """ON_TIMEOUT fires after the specified duration."""
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_TIMEOUT, timeout_seconds=0.1),
            action=CallbackAction.STOP_PROCESS,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        dispatcher.start_timeout_watchers(100)
        await asyncio.sleep(0.2)

        assert cb.fire_count == 1
        mock_pm.kill_process.assert_awaited()

    @pytest.mark.asyncio
    async def test_timeout_cancelled_on_exit(
        self, dispatcher: HookDispatcher, mock_pm: MagicMock
    ) -> None:
        """ON_TIMEOUT watcher is cancelled when the process exits."""
        cb_timeout = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_TIMEOUT, timeout_seconds=1.0),
            action=CallbackAction.STOP_PROCESS,
        )
        cb_exit = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_EXIT),
            action=CallbackAction.STOP_PROCESS,
        )
        tracked = _make_tracked(callbacks=[cb_timeout, cb_exit])
        mock_pm.get_process.return_value = tracked

        dispatcher.start_timeout_watchers(100)
        await asyncio.sleep(0.05)

        # Process exits before timeout
        await dispatcher._on_exit(100, 0)

        # Timeout callback should NOT have fired
        assert cb_timeout.fire_count == 0
        # Exit callback SHOULD have fired
        assert cb_exit.fire_count == 1


# ---------------------------------------------------------------------------
# NOTIFY_CHANNEL rate limiting
# ---------------------------------------------------------------------------


class TestNotifyRateLimit:
    @pytest.mark.asyncio
    async def test_notify_rate_limited(
        self, mock_pm: MagicMock
    ) -> None:
        """Rapid NOTIFY_CHANNEL fires are suppressed after the first."""
        notify_cb = AsyncMock()
        dispatcher = HookDispatcher(
            process_manager=mock_pm,
            notify_callback=notify_cb,
            default_output_delay=0.0,
        )
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_OUTPUT_MATCH, pattern=r"ERROR"),
            action=CallbackAction.NOTIFY_CHANNEL,
            context_message="Error detected",
            max_fires=0,
            min_message_interval=60.0,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        # Fire 5 times rapidly — only 1st should go through
        for _ in range(5):
            await dispatcher._on_line(100, "stdout", "ERROR: boom")

        assert cb.fire_count == 1
        assert notify_cb.await_count == 1
        assert cb._skipped_fires == 4

    @pytest.mark.asyncio
    async def test_notify_skipped_count_in_next_message(
        self, mock_pm: MagicMock
    ) -> None:
        """After cooldown, the next notification includes skipped count."""
        notify_cb = AsyncMock()
        dispatcher = HookDispatcher(
            process_manager=mock_pm,
            notify_callback=notify_cb,
            default_output_delay=0.0,
        )
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_OUTPUT_MATCH, pattern=r"ERROR"),
            action=CallbackAction.NOTIFY_CHANNEL,
            context_message="Error detected",
            max_fires=0,
            min_message_interval=60.0,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        # First fire goes through
        await dispatcher._on_line(100, "stdout", "ERROR: first")
        assert cb.fire_count == 1

        # Suppress 3 more
        for _ in range(3):
            await dispatcher._on_line(100, "stdout", "ERROR: repeated")
        assert cb._skipped_fires == 3

        # Simulate cooldown elapsed by rewinding _last_notify_time
        cb._last_notify_time = time.monotonic() - 120.0

        # Next fire should go through and mention 3 suppressed
        await dispatcher._on_line(100, "stdout", "ERROR: after cooldown")
        assert cb.fire_count == 2
        assert cb._skipped_fires == 0

        # Check that the notification context includes suppressed count
        last_call = notify_cb.call_args_list[-1]
        context_arg = last_call[0][1]
        assert "3 notification(s) suppressed" in context_arg

    @pytest.mark.asyncio
    async def test_rate_limit_does_not_affect_other_actions(
        self, mock_pm: MagicMock
    ) -> None:
        """STOP_PROCESS and other actions are not rate-limited."""
        dispatcher = HookDispatcher(
            process_manager=mock_pm,
            default_output_delay=0.0,
        )
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_OUTPUT_MATCH, pattern=r"ERROR"),
            action=CallbackAction.STOP_PROCESS,
            max_fires=0,
            min_message_interval=60.0,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        for _ in range(5):
            await dispatcher._on_line(100, "stdout", "ERROR: boom")

        # All 5 should fire — rate limit only applies to NOTIFY_CHANNEL
        assert cb.fire_count == 5
        assert mock_pm.kill_process.await_count == 5

    @pytest.mark.asyncio
    async def test_notify_rate_limit_zero_disables(
        self, mock_pm: MagicMock
    ) -> None:
        """min_message_interval=0 disables rate limiting."""
        notify_cb = AsyncMock()
        dispatcher = HookDispatcher(
            process_manager=mock_pm,
            notify_callback=notify_cb,
            default_output_delay=0.0,
        )
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_OUTPUT_MATCH, pattern=r"ERROR"),
            action=CallbackAction.NOTIFY_CHANNEL,
            max_fires=0,
            min_message_interval=0,
        )
        tracked = _make_tracked(callbacks=[cb])
        mock_pm.get_process.return_value = tracked

        for _ in range(5):
            await dispatcher._on_line(100, "stdout", "ERROR: boom")

        # All 5 should fire — rate limiting disabled
        assert cb.fire_count == 5
        assert notify_cb.await_count == 5
