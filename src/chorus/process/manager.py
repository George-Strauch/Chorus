"""ProcessManager — central singleton for all process lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chorus.process.models import (
    ProcessCallback,
    ProcessStatus,
    ProcessType,
    TrackedProcess,
)
from chorus.process.monitor import OutputMonitor
from chorus.tools.bash import _sanitized_env

if TYPE_CHECKING:
    from chorus.storage.db import Database

logger = logging.getLogger("chorus.process.manager")

# Cache stdbuf availability at import time (it's a static binary path)
_STDBUF_PATH: str | None = shutil.which("stdbuf")

_DEFAULT_SIGTERM_GRACE = 5.0


def _wrap_with_stdbuf(command: str) -> str:
    """Wrap *command* with ``stdbuf -oL`` to force line-buffered stdout.

    Most programs use full buffering (~4-8KB) when stdout is a pipe
    (not a TTY). This means output only reaches our OutputMonitor when
    the buffer fills or the process exits — breaking ON_OUTPUT_MATCH
    hooks that need to see lines in real time.

    ``stdbuf -oL`` uses LD_PRELOAD to override libc's buffering and
    force line-buffering on stdout. This works for most dynamically
    linked programs. It does NOT work for:
    - Statically linked binaries (Go, Rust, musl-linked)
    - Programs that explicitly call setvbuf() after startup
    - setuid binaries (LD_PRELOAD is ignored)

    For Python subprocesses, PYTHONUNBUFFERED=1 (set in _sanitized_env)
    is more reliable since it bypasses libc entirely.
    """
    if _STDBUF_PATH is None:
        return command
    return f"stdbuf -oL {command}"


class ProcessManager:
    """Central manager for tracked subprocess lifecycle.

    Parameters
    ----------
    chorus_home:
        Base directory for agent data (~/.chorus-agents/).
    db:
        Database instance for process persistence.
    host_execution:
        Whether to use the full host environment for subprocesses.
    """

    def __init__(
        self,
        chorus_home: Path,
        db: Database | None = None,
        host_execution: bool = False,
    ) -> None:
        self._chorus_home = chorus_home
        self._db = db
        self._host_execution = host_execution
        self._processes: dict[int, TrackedProcess] = {}
        self._monitors: dict[int, OutputMonitor] = {}
        self._subprocess_handles: dict[int, asyncio.subprocess.Process] = {}
        self._lock = asyncio.Lock()
        # Hook callbacks, wired by HookDispatcher
        self._on_line_callback: Any = None
        self._on_exit_callback: Any = None
        self._on_spawn_callback: Any = None

    def set_callbacks(
        self,
        on_line: Any = None,
        on_exit: Any = None,
        on_spawn: Any = None,
    ) -> None:
        """Set callbacks for hook integration (wired by HookDispatcher)."""
        self._on_line_callback = on_line
        self._on_exit_callback = on_exit
        self._on_spawn_callback = on_spawn

    async def spawn(
        self,
        command: str,
        workspace: Path,
        agent_name: str,
        process_type: ProcessType,
        callbacks: list[ProcessCallback] | None = None,
        context: str = "",
        model_for_hooks: str | None = None,
        hook_recursion_depth: int = 0,
        spawned_by_branch: int | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> TrackedProcess:
        """Spawn a subprocess and begin monitoring it.

        Returns a TrackedProcess with the assigned PID.
        """
        env = _sanitized_env(workspace, env_overrides, host_execution=self._host_execution)

        # Wrap with stdbuf to force line-buffered stdout so hooks see
        # output in real time instead of only at process exit.
        wrapped_command = _wrap_with_stdbuf(command)

        process = await asyncio.create_subprocess_shell(
            wrapped_command,
            cwd=workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        pid = process.pid
        if pid is None:
            raise RuntimeError("Failed to get PID from subprocess")

        now = datetime.now(UTC).isoformat()
        log_dir = self._chorus_home / "agents" / agent_name / "processes" / str(pid)

        rolling_tail: deque[str] = deque(maxlen=100)

        tracked = TrackedProcess(
            pid=pid,
            command=command,
            working_directory=str(workspace),
            agent_name=agent_name,
            started_at=now,
            process_type=process_type,
            spawned_by_branch=spawned_by_branch,
            callbacks=callbacks or [],
            context=context,
            rolling_tail=rolling_tail,
            model_for_hooks=model_for_hooks,
            hook_recursion_depth=hook_recursion_depth,
        )

        if callbacks and self._on_line_callback is None:
            logger.warning(
                "Spawning process with %d callback(s) but on_line_callback is None "
                "(HookDispatcher not wired?) — hooks will NOT fire",
                len(callbacks),
            )

        # Register tracked process BEFORE starting monitor so that
        # _on_line / _on_exit callbacks can find it via get_process().
        async with self._lock:
            self._processes[pid] = tracked
            self._subprocess_handles[pid] = process

        # Create and start monitor
        monitor = OutputMonitor(
            pid=pid,
            process=process,
            log_dir=log_dir,
            rolling_tail=rolling_tail,
            on_line=self._on_line_callback,
            on_exit=self._make_exit_handler(pid),
        )
        monitor.start()

        tracked.stdout_log = monitor.stdout_log
        tracked.stderr_log = monitor.stderr_log

        async with self._lock:
            self._monitors[pid] = monitor

        # Persist to DB
        await self._persist_process(tracked)

        logger.info(
            "Spawned %s process pid=%d cmd=%r for agent %s (%d callbacks, on_line=%s)",
            process_type.value, pid, command, agent_name,
            len(callbacks or []),
            "wired" if self._on_line_callback is not None else "NONE",
        )

        # Fire on_spawn callback (for timeout watchers etc.)
        if self._on_spawn_callback is not None:
            try:
                self._on_spawn_callback(pid)
            except Exception:
                logger.warning("on_spawn callback error for pid %d", pid, exc_info=True)

        return tracked

    def _make_exit_handler(self, pid: int) -> Any:
        """Create an exit callback closure for a specific PID."""
        async def _on_exit(p: int, exit_code: int | None) -> None:
            async with self._lock:
                tracked = self._processes.get(p)
                if tracked is None:
                    return
                if tracked.status == ProcessStatus.KILLED:
                    # Already killed — don't overwrite status
                    tracked.exit_code = exit_code
                else:
                    tracked.status = ProcessStatus.EXITED
                    tracked.exit_code = exit_code

                # Update log paths from monitor
                monitor = self._monitors.get(p)
                if monitor is not None:
                    tracked.stdout_log = monitor.stdout_log
                    tracked.stderr_log = monitor.stderr_log

            # Persist updated status
            await self._update_process_status(p, tracked.status, exit_code)

            # Fire external exit callback (hooks)
            if self._on_exit_callback is not None:
                try:
                    await self._on_exit_callback(p, exit_code)
                except Exception:
                    logger.warning("External exit callback error for pid %d", p, exc_info=True)

            logger.info("Process pid=%d exited with code %s", p, exit_code)

        return _on_exit

    async def kill_process(
        self,
        pid: int,
        grace_seconds: float = _DEFAULT_SIGTERM_GRACE,
    ) -> bool:
        """Kill a tracked process (SIGTERM → wait → SIGKILL).

        Returns True if the process was found and killed.
        """
        async with self._lock:
            tracked = self._processes.get(pid)
            proc = self._subprocess_handles.get(pid)
        if tracked is None or proc is None:
            return False

        if proc.returncode is not None:
            # Already exited
            return False

        # SIGTERM
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.terminate()

        # Wait for grace period
        try:
            await asyncio.wait_for(proc.wait(), timeout=grace_seconds)
        except TimeoutError:
            # SIGKILL
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()

        async with self._lock:
            tracked.status = ProcessStatus.KILLED
            tracked.exit_code = proc.returncode

        await self._update_process_status(pid, ProcessStatus.KILLED, proc.returncode)

        # Stop monitor
        monitor = self._monitors.get(pid)
        if monitor is not None:
            await monitor.stop()

        logger.info("Killed process pid=%d", pid)
        return True

    async def kill_all_for_agent(self, agent_name: str) -> int:
        """Kill all running processes for an agent. Returns count killed."""
        pids = [
            p.pid
            for p in self._processes.values()
            if p.agent_name == agent_name and p.status == ProcessStatus.RUNNING
        ]
        count = 0
        for pid in pids:
            if await self.kill_process(pid):
                count += 1
        return count

    def list_processes(self, agent_name: str | None = None) -> list[TrackedProcess]:
        """List tracked processes, optionally filtered by agent."""
        if agent_name is None:
            return list(self._processes.values())
        return [p for p in self._processes.values() if p.agent_name == agent_name]

    def get_process(self, pid: int) -> TrackedProcess | None:
        """Get a tracked process by PID."""
        return self._processes.get(pid)

    async def recover_on_startup(self) -> None:
        """Recover process state from DB on startup.

        For each process marked 'running' in the DB, check if the PID is
        still alive. Mark dead processes as LOST.
        """
        if self._db is None:
            return

        rows = await self._db.list_processes()
        recovered = 0
        lost = 0
        for row in rows:
            if row["status"] != "running":
                continue
            pid = row["pid"]
            if _is_pid_alive(pid):
                # Process is still running but we can't monitor it
                # (we lost the file descriptors). Mark as LOST.
                await self._db.update_process_status(pid, "lost", None)
                lost += 1
            else:
                await self._db.update_process_status(pid, "lost", None)
                lost += 1
            recovered += 1

        if recovered:
            logger.info(
                "Process recovery: %d tracked, %d marked lost", recovered, lost
            )

    async def add_callbacks(
        self, pid: int, callbacks: list[ProcessCallback]
    ) -> TrackedProcess | None:
        """Add callbacks to a running process.

        Returns the TrackedProcess if found and still running, None otherwise.
        """
        async with self._lock:
            tracked = self._processes.get(pid)
            if tracked is None or tracked.status != ProcessStatus.RUNNING:
                return None
            tracked.callbacks.extend(callbacks)

        await self._persist_callbacks(tracked)
        return tracked

    async def _persist_callbacks(self, tracked: TrackedProcess) -> None:
        """Serialize all callbacks and update the DB."""
        if self._db is None:
            return
        callbacks_json = json.dumps([cb.to_dict() for cb in tracked.callbacks])
        await self._db.update_process_callbacks(tracked.pid, callbacks_json)

    # ── DB persistence ──────────────────────────────────────────────

    async def _persist_process(self, tracked: TrackedProcess) -> None:
        if self._db is None:
            return
        callbacks_json = json.dumps([cb.to_dict() for cb in tracked.callbacks])
        await self._db.insert_process(
            pid=tracked.pid,
            command=tracked.command,
            working_directory=tracked.working_directory,
            agent_name=tracked.agent_name,
            started_at=tracked.started_at,
            process_type=tracked.process_type.value,
            spawned_by_branch=tracked.spawned_by_branch,
            stdout_log=tracked.stdout_log,
            stderr_log=tracked.stderr_log,
            status=tracked.status.value,
            callbacks_json=callbacks_json,
            context_json=tracked.context,
            model_for_hooks=tracked.model_for_hooks,
            hook_recursion_depth=tracked.hook_recursion_depth,
            discord_message_id=tracked.discord_message_id,
        )

    async def _update_process_status(
        self, pid: int, status: ProcessStatus, exit_code: int | None
    ) -> None:
        if self._db is None:
            return
        await self._db.update_process_status(pid, status.value, exit_code)


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
