"""Execution thread infrastructure for concurrent agent task handling."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chorus.storage.db import Database

logger = logging.getLogger("chorus.agent.threads")


class ThreadStatus(Enum):
    """Status of an execution thread."""

    RUNNING = "running"
    WAITING_FOR_PERMISSION = "waiting_for_permission"
    IDLE = "idle"
    COMPLETED = "completed"


@dataclass
class ThreadStep:
    """A single step in a thread's execution history."""

    step_number: int
    description: str
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: int | None = None


@dataclass
class ThreadMetrics:
    """Timing and step tracking for a thread."""

    created_at: datetime
    step_number: int = 0
    current_step: str = "Starting"
    step_history: list[ThreadStep] = field(default_factory=list)

    @property
    def elapsed_ms(self) -> int:
        """Wall-clock time since thread creation in milliseconds."""
        return int((datetime.now(UTC) - self.created_at).total_seconds() * 1000)

    def begin_step(self, description: str) -> ThreadStep:
        """End the current step (if any) and start a new one."""
        now = datetime.now(UTC)
        if self.step_history and self.step_history[-1].ended_at is None:
            prev = self.step_history[-1]
            prev.ended_at = now
            prev.duration_ms = int((prev.ended_at - prev.started_at).total_seconds() * 1000)
        self.step_number += 1
        self.current_step = description
        step = ThreadStep(
            step_number=self.step_number,
            description=description,
            started_at=now,
        )
        self.step_history.append(step)
        return step

    def finalize(self) -> None:
        """Close the last open step."""
        if self.step_history and self.step_history[-1].ended_at is None:
            now = datetime.now(UTC)
            last = self.step_history[-1]
            last.ended_at = now
            last.duration_ms = int((now - last.started_at).total_seconds() * 1000)


@dataclass
class ExecutionThread:
    """A single execution thread within an agent."""

    id: int
    agent_name: str
    messages: list[dict[str, Any]]
    status: ThreadStatus
    task: asyncio.Task[None] | None = None
    discord_message_ids: list[int] = field(default_factory=list)
    metrics: ThreadMetrics = field(
        default_factory=lambda: ThreadMetrics(created_at=datetime.now(UTC))
    )
    summary: str | None = None
    completed_at: datetime | None = None


# Type alias for the runner callback
ThreadRunner = Callable[[ExecutionThread], Awaitable[None]]


async def _default_runner(thread: ExecutionThread) -> None:
    """No-op runner — sets status to COMPLETED."""
    thread.status = ThreadStatus.COMPLETED
    thread.metrics.finalize()


class ThreadManager:
    """Manages concurrent execution threads for a single agent."""

    def __init__(
        self,
        agent_name: str,
        db: Database | None = None,
        cleanup_after_seconds: float = 600,
    ) -> None:
        self._agent_name = agent_name
        self._db = db
        self._cleanup_after = cleanup_after_seconds
        self._threads: dict[int, ExecutionThread] = {}
        self._message_to_thread: dict[int, int] = {}  # discord_msg_id -> thread_id
        self._file_locks: dict[str, asyncio.Lock] = {}
        self._thread_file_locks: dict[int, set[str]] = {}  # thread_id -> set of locked paths
        self._next_id: int = 1

    def create_thread(self, initial_message: dict[str, Any]) -> ExecutionThread:
        """Create a new execution thread with an initial message."""
        thread = ExecutionThread(
            id=self._next_id,
            agent_name=self._agent_name,
            messages=[initial_message],
            status=ThreadStatus.IDLE,
            metrics=ThreadMetrics(created_at=datetime.now(UTC)),
        )
        self._threads[thread.id] = thread
        self._next_id += 1
        logger.info("Created thread #%d for agent %s", thread.id, self._agent_name)
        return thread

    def start_thread(
        self,
        thread: ExecutionThread,
        runner: ThreadRunner | None = None,
    ) -> None:
        """Start a thread's execution via asyncio.Task."""
        actual_runner = runner or _default_runner
        thread.status = ThreadStatus.RUNNING

        async def _wrapped() -> None:
            try:
                await actual_runner(thread)
            except asyncio.CancelledError:
                logger.info("Thread #%d cancelled", thread.id)
            except Exception:
                logger.exception("Thread #%d failed", thread.id)
            finally:
                thread.status = ThreadStatus.COMPLETED
                thread.completed_at = datetime.now(UTC)
                thread.metrics.finalize()
                self._release_all_file_locks(thread.id)

        thread.task = asyncio.create_task(_wrapped())

    def route_message(self, discord_message_id: int) -> ExecutionThread | None:
        """Look up the thread that produced a given Discord message."""
        thread_id = self._message_to_thread.get(discord_message_id)
        if thread_id is None:
            return None
        return self._threads.get(thread_id)

    def register_bot_message(self, msg_id: int, thread_id: int) -> None:
        """Map a Discord message ID to a thread for reply routing."""
        self._message_to_thread[msg_id] = thread_id
        thread = self._threads.get(thread_id)
        if thread is not None:
            thread.discord_message_ids.append(msg_id)

    def get_thread(self, thread_id: int) -> ExecutionThread | None:
        """Get a thread by ID."""
        return self._threads.get(thread_id)

    async def kill_thread(self, thread_id: int) -> bool:
        """Cancel a running thread's task and mark it completed."""
        thread = self._threads.get(thread_id)
        if thread is None:
            return False
        if thread.task is not None and not thread.task.done():
            thread.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await thread.task
        thread.status = ThreadStatus.COMPLETED
        thread.completed_at = datetime.now(UTC)
        thread.metrics.finalize()
        self._release_all_file_locks(thread_id)
        return True

    async def kill_all(self) -> int:
        """Kill all active (non-completed) threads. Returns count killed."""
        active_ids = [
            t.id for t in self._threads.values() if t.status != ThreadStatus.COMPLETED
        ]
        for tid in active_ids:
            await self.kill_thread(tid)
        return len(active_ids)

    def list_active(self) -> list[ExecutionThread]:
        """Return all non-completed threads."""
        return [t for t in self._threads.values() if t.status != ThreadStatus.COMPLETED]

    def list_all(self) -> list[ExecutionThread]:
        """Return all threads."""
        return list(self._threads.values())

    def cleanup_completed(self) -> None:
        """Remove completed threads older than the cleanup window."""
        now = datetime.now(UTC)
        to_remove = []
        for tid, thread in self._threads.items():
            if (
                thread.status == ThreadStatus.COMPLETED
                and thread.completed_at is not None
                and (now - thread.completed_at).total_seconds() > self._cleanup_after
            ):
                to_remove.append(tid)
        for tid in to_remove:
            del self._threads[tid]
        if to_remove:
            logger.info(
                "Cleaned up %d completed thread(s) for agent %s",
                len(to_remove),
                self._agent_name,
            )

    # ── File Locking ─────────────────────────────────────────────────────

    async def acquire_file_lock(self, path: str, timeout: float = 30) -> bool:
        """Acquire a per-path lock. Returns False on timeout."""
        lock = self._file_locks.setdefault(path, asyncio.Lock())
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    def release_file_lock(self, path: str) -> None:
        """Release a file lock."""
        lock = self._file_locks.get(path)
        if lock is not None and lock.locked():
            lock.release()

    def is_file_locked(self, path: str) -> bool:
        """Check if a path is currently locked."""
        lock = self._file_locks.get(path)
        return lock is not None and lock.locked()

    def _release_all_file_locks(self, thread_id: int) -> None:
        """Release all file locks held by a thread."""
        # Release all locks that are currently held
        # This is a safety net — in practice, individual tools should release their locks
        for _path, lock in list(self._file_locks.items()):
            if lock.locked():
                with contextlib.suppress(RuntimeError):
                    lock.release()

    # ── Step Persistence ─────────────────────────────────────────────────

    async def persist_step(self, thread: ExecutionThread, step: ThreadStep) -> None:
        """Write a completed step to SQLite if DB is available."""
        if self._db is None:
            return
        await self._db.persist_thread_step(
            agent_name=thread.agent_name,
            thread_id=thread.id,
            step_number=step.step_number,
            description=step.description,
            started_at=step.started_at.isoformat(),
            ended_at=step.ended_at.isoformat() if step.ended_at else None,
            duration_ms=step.duration_ms,
        )


def build_thread_status(thread_manager: ThreadManager, current_thread_id: int) -> str:
    """Format active thread info for system prompt injection."""
    threads = thread_manager.list_all()
    if not threads:
        return "No active threads."
    lines = ["Active threads:"]
    for t in threads:
        if t.status == ThreadStatus.COMPLETED:
            continue
        marker = " (this thread)" if t.id == current_thread_id else ""
        elapsed_s = t.metrics.elapsed_ms / 1000
        summary = t.summary or "Starting..."
        lines.append(
            f"  #{t.id}{marker}: {summary} "
            f"— step {t.metrics.step_number}, {elapsed_s:.0f}s elapsed, "
            f"currently: {t.metrics.current_step} [{t.status.value}]"
        )
    if len(lines) == 1:
        return "No active threads."
    return "\n".join(lines)
