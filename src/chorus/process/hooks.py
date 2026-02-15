"""HookDispatcher — evaluates callbacks and executes actions on process events."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, Protocol

from chorus.process.models import (
    CallbackAction,
    ExitFilter,
    ProcessCallback,
    ProcessStatus,
    TriggerType,
)

if TYPE_CHECKING:
    from chorus.process.manager import ProcessManager

logger = logging.getLogger("chorus.process.hooks")


# ---------------------------------------------------------------------------
# BranchSpawner protocol
# ---------------------------------------------------------------------------


class BranchSpawner(Protocol):
    """Protocol for spawning hook-triggered branches (avoids circular imports)."""

    async def spawn_hook_branch(
        self,
        agent_name: str,
        hook_context: str,
        model: str | None = None,
        recursion_depth: int = 0,
    ) -> None: ...


# ---------------------------------------------------------------------------
# HookDispatcher
# ---------------------------------------------------------------------------


class HookDispatcher:
    """Evaluates process callbacks and dispatches actions.

    Wired into ProcessManager's on_line/on_exit/on_spawn callbacks.

    Parameters
    ----------
    process_manager:
        Reference to the ProcessManager for STOP_PROCESS actions.
    branch_spawner:
        Callback for SPAWN_BRANCH actions.
    thread_kill_callback:
        Callback for STOP_BRANCH actions — ``(agent_name, branch_id) → None``.
    inject_callback:
        Callback for INJECT_CONTEXT actions — ``(agent_name, branch_id, message) → None``.
    notify_callback:
        Callback for NOTIFY_CHANNEL actions — ``(agent_name, context, tracked) → None``.
    default_output_delay:
        Default delay in seconds for output match callbacks.
    max_recursion_depth:
        Maximum hook recursion depth.
    """

    def __init__(
        self,
        process_manager: ProcessManager,
        branch_spawner: BranchSpawner | None = None,
        thread_kill_callback: Callable[..., Coroutine[Any, Any, None]] | None = None,
        inject_callback: Callable[..., Coroutine[Any, Any, None]] | None = None,
        notify_callback: Callable[..., Coroutine[Any, Any, None]] | None = None,
        default_output_delay: float = 2.0,
        max_recursion_depth: int = 3,
    ) -> None:
        self._pm = process_manager
        self._branch_spawner = branch_spawner
        self._thread_kill = thread_kill_callback
        self._inject = inject_callback
        self._notify = notify_callback
        self._default_output_delay = default_output_delay
        self._max_recursion_depth = max_recursion_depth
        self._spawn_semaphore = asyncio.Semaphore(3)
        self._pending_delays: dict[int, asyncio.Task[None]] = {}
        self._timeout_tasks: dict[int, asyncio.Task[None]] = {}

    def wire_to_manager(self) -> None:
        """Connect this dispatcher to the ProcessManager's callbacks."""
        self._pm.set_callbacks(
            on_line=self._on_line,
            on_exit=self._on_exit,
            on_spawn=self._on_spawn,
        )

    def _on_spawn(self, pid: int) -> None:
        """Called when a process is spawned — starts timeout watchers."""
        self.start_timeout_watchers(pid)

    # ── Start/stop timeout watchers ─────────────────────────────────────

    def start_timeout_watchers(self, pid: int) -> None:
        """Start ON_TIMEOUT watchers for a process's callbacks."""
        tracked = self._pm.get_process(pid)
        if tracked is None:
            return
        for cb in tracked.callbacks:
            if (
                cb.trigger.type == TriggerType.ON_TIMEOUT
                and cb.trigger.timeout_seconds is not None
                and not cb.exhausted
            ):
                task = asyncio.create_task(
                    self._timeout_watcher(pid, cb)
                )
                self._timeout_tasks[pid] = task

    async def _timeout_watcher(self, pid: int, cb: ProcessCallback) -> None:
        """Wait for the timeout duration and then fire the callback."""
        assert cb.trigger.timeout_seconds is not None
        try:
            await asyncio.sleep(cb.trigger.timeout_seconds)
        except asyncio.CancelledError:
            return

        tracked = self._pm.get_process(pid)
        if tracked is None or tracked.status != ProcessStatus.RUNNING:
            return
        if cb.exhausted:
            return

        await self._fire_callback(pid, cb, "Process timed out")

    # ── Event handlers (wired to ProcessManager) ─────────────────────

    async def _on_line(self, pid: int, stream_name: str, line: str) -> None:
        """Called for each line of output — evaluates ON_OUTPUT_MATCH triggers."""
        tracked = self._pm.get_process(pid)
        if tracked is None:
            return

        for cb in tracked.callbacks:
            if cb.trigger.type != TriggerType.ON_OUTPUT_MATCH:
                continue
            if cb.exhausted:
                continue
            pattern = cb.trigger.compiled_pattern
            if pattern is None:
                continue
            if not pattern.search(line):
                continue

            # Matched! Apply output delay if configured
            delay = cb.output_delay_seconds or self._default_output_delay
            if delay > 0:
                await self._delayed_fire(pid, cb, line, delay)
            else:
                await self._fire_callback(pid, cb, f"Output matched: {line}")

    async def _on_exit(self, pid: int, exit_code: int | None) -> None:
        """Called when a process exits — evaluates ON_EXIT triggers."""
        # Cancel any pending timeout watchers
        timeout_task = self._timeout_tasks.pop(pid, None)
        if timeout_task is not None and not timeout_task.done():
            timeout_task.cancel()

        tracked = self._pm.get_process(pid)
        if tracked is None:
            return

        for cb in tracked.callbacks:
            if cb.trigger.type != TriggerType.ON_EXIT:
                continue
            if cb.exhausted:
                continue

            # Check exit filter
            ef = cb.trigger.exit_filter
            if ef == ExitFilter.SUCCESS and exit_code != 0:
                continue
            if ef == ExitFilter.FAILURE and exit_code == 0:
                continue

            context = (
                f"Process exited with code {exit_code}. "
                f"Command: {tracked.command}"
            )
            await self._fire_callback(pid, cb, context)

    # ── Delayed firing ──────────────────────────────────────────────────

    async def _delayed_fire(
        self, pid: int, cb: ProcessCallback, trigger_line: str, delay: float
    ) -> None:
        """Wait for delay, accumulating output, then fire."""
        # Cancel any existing delay for this callback
        key = id(cb)

        async def _wait_and_fire() -> None:
            await asyncio.sleep(delay)
            tracked = self._pm.get_process(pid)
            if tracked is None:
                return
            # Build context with accumulated output
            tail_lines = list(tracked.rolling_tail)
            recent = "\n".join(tail_lines[-20:]) if tail_lines else ""
            context = (
                f"Output matched pattern: {trigger_line}\n"
                f"Recent output after delay:\n{recent}"
            )
            await self._fire_callback(pid, cb, context)

        task = asyncio.create_task(_wait_and_fire())
        self._pending_delays[key] = task

    # ── Action dispatch ────────────────────────────────────────────────

    async def _fire_callback(
        self, pid: int, cb: ProcessCallback, context: str
    ) -> None:
        """Increment fire count and dispatch the action."""
        cb.fire_count += 1

        tracked = self._pm.get_process(pid)
        if tracked is None:
            return

        action = cb.action
        full_context = cb.context_message
        if full_context and context:
            full_context = f"{full_context}\n\n{context}"
        elif context:
            full_context = context

        logger.info(
            "Firing callback %s for pid %d (fire %d/%d)",
            action.value, pid, cb.fire_count, cb.max_fires,
        )

        if action == CallbackAction.STOP_PROCESS:
            await self._pm.kill_process(pid)

        elif action == CallbackAction.STOP_BRANCH:
            if tracked.spawned_by_branch is not None and self._thread_kill is not None:
                await self._thread_kill(tracked.agent_name, tracked.spawned_by_branch)

        elif action == CallbackAction.INJECT_CONTEXT:
            if tracked.spawned_by_branch is not None and self._inject is not None:
                await self._inject(
                    tracked.agent_name,
                    tracked.spawned_by_branch,
                    full_context,
                )

        elif action == CallbackAction.NOTIFY_CHANNEL:
            if self._notify is not None:
                await self._notify(tracked.agent_name, full_context, tracked)

        elif action == CallbackAction.SPAWN_BRANCH:
            if tracked.hook_recursion_depth >= self._max_recursion_depth:
                logger.warning(
                    "Hook recursion depth exceeded for pid %d (depth %d)",
                    pid, tracked.hook_recursion_depth,
                )
                return

            if self._branch_spawner is None:
                logger.warning("No branch spawner configured for SPAWN_BRANCH action")
                return

            # Rate limit spawning
            async with self._spawn_semaphore:
                # Build rich context for the spawned branch
                tail_lines = list(tracked.rolling_tail)
                recent_output = "\n".join(tail_lines[-30:]) if tail_lines else "(no output)"
                hook_context = (
                    f"A process hook was triggered.\n\n"
                    f"**Process:** PID {pid}\n"
                    f"**Command:** `{tracked.command}`\n"
                    f"**Status:** {tracked.status.value}"
                    f"{f' (exit {tracked.exit_code})' if tracked.exit_code is not None else ''}\n"
                    f"**Trigger context:** {full_context}\n\n"
                    f"**Recent output:**\n```\n{recent_output}\n```\n\n"
                    f"Respond to this event as instructed."
                )
                await self._branch_spawner.spawn_hook_branch(
                    agent_name=tracked.agent_name,
                    hook_context=hook_context,
                    model=tracked.model_for_hooks,
                    recursion_depth=tracked.hook_recursion_depth + 1,
                )
