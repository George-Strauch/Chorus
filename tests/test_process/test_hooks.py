"""Tests for HookDispatcher — trigger evaluation, action dispatch, safety."""

from __future__ import annotations

import asyncio
from collections import deque
from unittest.mock import AsyncMock, MagicMock

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


# ---------------------------------------------------------------------------
# ON_TIMEOUT triggers
# ---------------------------------------------------------------------------


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
