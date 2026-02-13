"""Tests for chorus.agent.threads — execution thread infrastructure."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from chorus.agent.threads import (
    ExecutionThread,
    ThreadManager,
    ThreadMetrics,
    ThreadStatus,
    build_thread_status,
)

# ── Lifecycle ────────────────────────────────────────────────────────────────


class TestThreadLifecycle:
    def test_create_thread(self, thread_manager: ThreadManager) -> None:
        thread = thread_manager.create_thread({"role": "user", "content": "hello"})
        assert thread.id == 1
        assert thread.agent_name == "test-agent"
        assert thread.status == ThreadStatus.IDLE
        assert len(thread.messages) == 1
        assert thread.messages[0]["content"] == "hello"

    async def test_route_to_existing(self, thread_manager: ThreadManager) -> None:
        thread = thread_manager.create_thread({"role": "user", "content": "hello"})
        thread_manager.register_bot_message(9999, thread.id)
        routed = thread_manager.route_message(9999)
        assert routed is not None
        assert routed.id == thread.id

    def test_route_unknown_returns_none(self, thread_manager: ThreadManager) -> None:
        result = thread_manager.route_message(99999)
        assert result is None

    async def test_completes_on_finish(self, thread_manager: ThreadManager) -> None:
        thread = thread_manager.create_thread({"role": "user", "content": "hello"})

        async def runner(t: ExecutionThread) -> None:
            t.status = ThreadStatus.COMPLETED

        thread_manager.start_thread(thread, runner=runner)
        assert thread.task is not None
        await thread.task
        assert thread.status == ThreadStatus.COMPLETED

    async def test_kill_cancels_task(self, thread_manager: ThreadManager) -> None:
        thread = thread_manager.create_thread({"role": "user", "content": "hello"})

        async def slow_runner(t: ExecutionThread) -> None:
            await asyncio.sleep(100)

        thread_manager.start_thread(thread, runner=slow_runner)
        assert thread.task is not None
        killed = await thread_manager.kill_thread(thread.id)
        assert killed is True
        assert thread.status == ThreadStatus.COMPLETED
        assert thread.completed_at is not None

    async def test_kill_all(self, thread_manager: ThreadManager) -> None:
        async def slow_runner(t: ExecutionThread) -> None:
            await asyncio.sleep(100)

        t1 = thread_manager.create_thread({"role": "user", "content": "a"})
        t2 = thread_manager.create_thread({"role": "user", "content": "b"})
        thread_manager.start_thread(t1, runner=slow_runner)
        thread_manager.start_thread(t2, runner=slow_runner)
        count = await thread_manager.kill_all()
        assert count == 2
        assert t1.status == ThreadStatus.COMPLETED
        assert t2.status == ThreadStatus.COMPLETED

    async def test_cleanup_after_timeout(self, thread_manager: ThreadManager) -> None:
        thread = thread_manager.create_thread({"role": "user", "content": "hello"})
        thread.status = ThreadStatus.COMPLETED
        # Set completed_at far in the past
        thread.completed_at = datetime(2020, 1, 1, tzinfo=UTC)
        thread_manager.cleanup_completed()
        assert thread_manager.get_thread(thread.id) is None


# ── Concurrency ──────────────────────────────────────────────────────────────


class TestConcurrency:
    async def test_two_threads_concurrent(self, thread_manager: ThreadManager) -> None:
        order: list[str] = []

        async def runner_a(t: ExecutionThread) -> None:
            order.append("a_start")
            await asyncio.sleep(0.05)
            order.append("a_end")
            t.status = ThreadStatus.COMPLETED

        async def runner_b(t: ExecutionThread) -> None:
            order.append("b_start")
            await asyncio.sleep(0.05)
            order.append("b_end")
            t.status = ThreadStatus.COMPLETED

        t1 = thread_manager.create_thread({"role": "user", "content": "a"})
        t2 = thread_manager.create_thread({"role": "user", "content": "b"})
        thread_manager.start_thread(t1, runner=runner_a)
        thread_manager.start_thread(t2, runner=runner_b)
        assert t1.task is not None
        assert t2.task is not None
        await asyncio.gather(t1.task, t2.task)
        # Both started before either ended
        assert order.index("a_start") < order.index("a_end")
        assert order.index("b_start") < order.index("b_end")
        # Interleaved start
        assert "a_start" in order[:2]
        assert "b_start" in order[:2]

    async def test_no_blocking(self, thread_manager: ThreadManager) -> None:
        t1 = thread_manager.create_thread({"role": "user", "content": "a"})

        async def slow_runner(t: ExecutionThread) -> None:
            await asyncio.sleep(100)

        thread_manager.start_thread(t1, runner=slow_runner)
        # Creating and starting a second thread should work immediately
        t2 = thread_manager.create_thread({"role": "user", "content": "b"})
        assert t2.id != t1.id
        await thread_manager.kill_all()

    async def test_permission_wait_no_block(self, thread_manager: ThreadManager) -> None:
        waited = asyncio.Event()

        async def perm_runner(t: ExecutionThread) -> None:
            t.status = ThreadStatus.WAITING_FOR_PERMISSION
            waited.set()
            await asyncio.sleep(100)

        async def fast_runner(t: ExecutionThread) -> None:
            await waited.wait()
            t.status = ThreadStatus.COMPLETED

        t1 = thread_manager.create_thread({"role": "user", "content": "a"})
        t2 = thread_manager.create_thread({"role": "user", "content": "b"})
        thread_manager.start_thread(t1, runner=perm_runner)
        thread_manager.start_thread(t2, runner=fast_runner)
        assert t2.task is not None
        await t2.task
        assert t2.status == ThreadStatus.COMPLETED
        assert t1.status == ThreadStatus.WAITING_FOR_PERMISSION
        await thread_manager.kill_all()


# ── File Locking ─────────────────────────────────────────────────────────────


class TestFileLocking:
    async def test_acquire(self, thread_manager: ThreadManager) -> None:
        acquired = await thread_manager.acquire_file_lock("main.py")
        assert acquired is True
        thread_manager.release_file_lock("main.py")

    async def test_blocks_same_file(self, thread_manager: ThreadManager) -> None:
        await thread_manager.acquire_file_lock("main.py")
        # Second acquire with short timeout should fail
        acquired = await thread_manager.acquire_file_lock("main.py", timeout=0.1)
        assert acquired is False
        thread_manager.release_file_lock("main.py")

    async def test_allows_different_files(self, thread_manager: ThreadManager) -> None:
        await thread_manager.acquire_file_lock("main.py")
        acquired = await thread_manager.acquire_file_lock("other.py", timeout=0.1)
        assert acquired is True
        thread_manager.release_file_lock("main.py")
        thread_manager.release_file_lock("other.py")

    async def test_allows_reads(self, thread_manager: ThreadManager) -> None:
        # is_file_locked returns info but reads don't acquire locks
        await thread_manager.acquire_file_lock("main.py")
        assert thread_manager.is_file_locked("main.py") is True
        assert thread_manager.is_file_locked("other.py") is False
        thread_manager.release_file_lock("main.py")

    async def test_released_on_completion(self, thread_manager: ThreadManager) -> None:
        await thread_manager.acquire_file_lock("main.py")
        thread_manager.release_file_lock("main.py")
        # Should be re-acquirable
        acquired = await thread_manager.acquire_file_lock("main.py", timeout=0.1)
        assert acquired is True
        thread_manager.release_file_lock("main.py")

    async def test_released_on_kill(self, thread_manager: ThreadManager) -> None:
        thread = thread_manager.create_thread({"role": "user", "content": "hello"})

        async def locking_runner(t: ExecutionThread) -> None:
            await thread_manager.acquire_file_lock("main.py")
            await asyncio.sleep(100)

        thread_manager.start_thread(thread, runner=locking_runner)
        # Give the runner a moment to acquire the lock
        await asyncio.sleep(0.05)
        assert thread_manager.is_file_locked("main.py") is True
        await thread_manager.kill_thread(thread.id)
        # After kill, lock should be released
        assert thread_manager.is_file_locked("main.py") is False

    async def test_timeout_returns_false(self, thread_manager: ThreadManager) -> None:
        await thread_manager.acquire_file_lock("main.py")
        result = await thread_manager.acquire_file_lock("main.py", timeout=0.05)
        assert result is False
        thread_manager.release_file_lock("main.py")


# ── Message Routing ──────────────────────────────────────────────────────────


class TestMessageRouting:
    def test_reference_maps_to_thread(self, thread_manager: ThreadManager) -> None:
        thread = thread_manager.create_thread({"role": "user", "content": "hello"})
        thread_manager.register_bot_message(5000, thread.id)
        result = thread_manager.route_message(5000)
        assert result is not None
        assert result.id == thread.id

    def test_non_reply_returns_none(self, thread_manager: ThreadManager) -> None:
        thread_manager.create_thread({"role": "user", "content": "hello"})
        result = thread_manager.route_message(99999)
        assert result is None

    def test_bot_msgs_tagged(self, thread_manager: ThreadManager) -> None:
        thread = thread_manager.create_thread({"role": "user", "content": "hello"})
        thread_manager.register_bot_message(5001, thread.id)
        thread_manager.register_bot_message(5002, thread.id)
        assert 5001 in thread.discord_message_ids
        assert 5002 in thread.discord_message_ids


# ── Metrics ──────────────────────────────────────────────────────────────────


class TestMetrics:
    def test_elapsed_time(self) -> None:
        metrics = ThreadMetrics(created_at=datetime.now(UTC))
        elapsed = metrics.elapsed_ms
        assert elapsed >= 0

    def test_increments_step(self) -> None:
        metrics = ThreadMetrics(created_at=datetime.now(UTC))
        metrics.begin_step("Step one")
        assert metrics.step_number == 1
        metrics.begin_step("Step two")
        assert metrics.step_number == 2

    def test_ends_previous_step(self) -> None:
        metrics = ThreadMetrics(created_at=datetime.now(UTC))
        metrics.begin_step("Step one")
        metrics.begin_step("Step two")
        assert metrics.step_history[0].ended_at is not None
        assert metrics.step_history[0].duration_ms is not None

    def test_updates_current_step(self) -> None:
        metrics = ThreadMetrics(created_at=datetime.now(UTC))
        metrics.begin_step("Calling LLM")
        assert metrics.current_step == "Calling LLM"
        metrics.begin_step("Running bash: pytest")
        assert metrics.current_step == "Running bash: pytest"

    def test_records_all_steps(self) -> None:
        metrics = ThreadMetrics(created_at=datetime.now(UTC))
        metrics.begin_step("One")
        metrics.begin_step("Two")
        metrics.begin_step("Three")
        assert len(metrics.step_history) == 3
        assert [s.description for s in metrics.step_history] == ["One", "Two", "Three"]

    def test_timestamp_and_duration(self) -> None:
        metrics = ThreadMetrics(created_at=datetime.now(UTC))
        metrics.begin_step("Step one")
        step = metrics.step_history[0]
        assert step.started_at is not None
        assert step.step_number == 1
        # Duration not set until ended
        assert step.duration_ms is None
        metrics.begin_step("Step two")
        # Now step one should have duration
        assert metrics.step_history[0].duration_ms is not None
        assert metrics.step_history[0].duration_ms >= 0

    async def test_persisted_to_sqlite(
        self, thread_manager_with_db: ThreadManager
    ) -> None:
        tm = thread_manager_with_db
        thread = tm.create_thread({"role": "user", "content": "hello"})
        thread.metrics.begin_step("Step one")
        thread.metrics.begin_step("Step two")  # Ends step one
        # Persist step one
        await tm.persist_step(thread, thread.metrics.step_history[0])
        assert tm._db is not None
        steps = await tm._db.get_thread_steps("test-agent", thread.id)
        assert len(steps) == 1
        assert steps[0]["description"] == "Step one"

    async def test_queryable_after_complete(
        self, thread_manager_with_db: ThreadManager
    ) -> None:
        tm = thread_manager_with_db
        thread = tm.create_thread({"role": "user", "content": "hello"})
        thread.metrics.begin_step("Step one")
        thread.metrics.begin_step("Step two")
        thread.metrics.finalize()

        for step in thread.metrics.step_history:
            await tm.persist_step(thread, step)

        assert tm._db is not None
        steps = await tm._db.get_thread_steps("test-agent", thread.id)
        assert len(steps) == 2
        assert steps[0]["step_number"] == 1
        assert steps[1]["step_number"] == 2

    def test_finalize_closes_step(self) -> None:
        metrics = ThreadMetrics(created_at=datetime.now(UTC))
        metrics.begin_step("Final step")
        metrics.finalize()
        assert metrics.step_history[-1].ended_at is not None
        assert metrics.step_history[-1].duration_ms is not None

    def test_created_at_set(self) -> None:
        before = datetime.now(UTC)
        metrics = ThreadMetrics(created_at=datetime.now(UTC))
        assert metrics.created_at >= before


# ── Context Injection ────────────────────────────────────────────────────────


class TestContextInjection:
    def test_shows_this_thread(self, thread_manager: ThreadManager) -> None:
        thread = thread_manager.create_thread({"role": "user", "content": "hello"})
        thread.summary = "Working on auth"
        thread.metrics.begin_step("Calling LLM")
        status = build_thread_status(thread_manager, thread.id)
        assert "(this thread)" in status
        assert "Working on auth" in status

    def test_shows_other_threads(self, thread_manager: ThreadManager) -> None:
        t1 = thread_manager.create_thread({"role": "user", "content": "a"})
        t2 = thread_manager.create_thread({"role": "user", "content": "b"})
        t1.summary = "Auth refactor"
        t2.summary = "Test writing"
        status = build_thread_status(thread_manager, t1.id)
        assert "Auth refactor" in status
        assert "Test writing" in status

    def test_includes_metrics(self, thread_manager: ThreadManager) -> None:
        thread = thread_manager.create_thread({"role": "user", "content": "hello"})
        thread.summary = "Working"
        thread.metrics.begin_step("Calling LLM")
        status = build_thread_status(thread_manager, thread.id)
        assert "step 1" in status
        assert "Calling LLM" in status


# ── Rate Limiting ────────────────────────────────────────────────────────────


class TestRateLimiting:
    async def test_respects_limit(self) -> None:
        from chorus.agent.message_queue import ChannelMessageQueue

        mock_channel = AsyncMock()
        mock_channel.send = AsyncMock(return_value=AsyncMock())
        queue = ChannelMessageQueue(mock_channel, max_per_window=3, window_seconds=1.0)
        # Should be able to send 3 quickly
        for i in range(3):
            await queue.send(f"msg {i}")
        assert mock_channel.send.call_count == 3

    async def test_fifo_order(self) -> None:
        from chorus.agent.message_queue import ChannelMessageQueue

        sent: list[str] = []
        mock_channel = AsyncMock()

        async def track_send(content: str, **kwargs: object) -> AsyncMock:
            sent.append(content)
            return AsyncMock()

        mock_channel.send = track_send
        queue = ChannelMessageQueue(mock_channel, max_per_window=10, window_seconds=1.0)
        await queue.send("first")
        await queue.send("second")
        await queue.send("third")
        assert sent == ["first", "second", "third"]


# ── Thread IDs ───────────────────────────────────────────────────────────────


class TestThreadIds:
    def test_auto_increment(self, thread_manager: ThreadManager) -> None:
        t1 = thread_manager.create_thread({"role": "user", "content": "a"})
        t2 = thread_manager.create_thread({"role": "user", "content": "b"})
        assert t1.id == 1
        assert t2.id == 2

    def test_unique_per_manager(self) -> None:
        tm1 = ThreadManager("agent-1")
        tm2 = ThreadManager("agent-2")
        t1 = tm1.create_thread({"role": "user", "content": "a"})
        t2 = tm2.create_thread({"role": "user", "content": "b"})
        # Both start at 1 — IDs are per-manager
        assert t1.id == 1
        assert t2.id == 1


# ── List Methods ─────────────────────────────────────────────────────────────


class TestListMethods:
    def test_list_active_excludes_completed(self, thread_manager: ThreadManager) -> None:
        t1 = thread_manager.create_thread({"role": "user", "content": "a"})
        t2 = thread_manager.create_thread({"role": "user", "content": "b"})
        t1.status = ThreadStatus.COMPLETED
        active = thread_manager.list_active()
        assert len(active) == 1
        assert active[0].id == t2.id
        all_threads = thread_manager.list_all()
        assert len(all_threads) == 2


# ── Main Thread ─────────────────────────────────────────────────────────────


class TestMainThread:
    def test_no_main_thread_initially(self, thread_manager: ThreadManager) -> None:
        assert thread_manager.get_main_thread() is None

    def test_set_main_thread(self, thread_manager: ThreadManager) -> None:
        thread = thread_manager.create_thread({"role": "user", "content": "hello"})
        thread_manager.set_main_thread(thread.id)
        assert thread_manager.get_main_thread() is thread

    def test_set_main_thread_invalid_raises(self, thread_manager: ThreadManager) -> None:
        with pytest.raises(ValueError, match="Unknown thread"):
            thread_manager.set_main_thread(999)

    def test_break_main_thread(self, thread_manager: ThreadManager) -> None:
        thread = thread_manager.create_thread({"role": "user", "content": "hello"})
        thread_manager.set_main_thread(thread.id)
        thread_manager.break_main_thread()
        assert thread_manager.get_main_thread() is None

    def test_break_when_no_main_is_noop(self, thread_manager: ThreadManager) -> None:
        thread_manager.break_main_thread()  # should not raise
        assert thread_manager.get_main_thread() is None

    def test_main_thread_survives_new_threads(self, thread_manager: ThreadManager) -> None:
        t1 = thread_manager.create_thread({"role": "user", "content": "a"})
        thread_manager.set_main_thread(t1.id)
        thread_manager.create_thread({"role": "user", "content": "b"})
        thread_manager.create_thread({"role": "user", "content": "c"})
        assert thread_manager.get_main_thread() is t1

    def test_create_thread_with_is_main(self, thread_manager: ThreadManager) -> None:
        thread = thread_manager.create_thread(
            {"role": "user", "content": "hello"}, is_main=True
        )
        assert thread_manager.get_main_thread() is thread

    def test_get_main_thread_after_cleanup(self, thread_manager: ThreadManager) -> None:
        thread = thread_manager.create_thread(
            {"role": "user", "content": "hello"}, is_main=True
        )
        thread.status = ThreadStatus.COMPLETED
        thread.completed_at = datetime(2020, 1, 1, tzinfo=UTC)
        thread_manager.cleanup_completed()
        # Thread was cleaned up — main should reset
        assert thread_manager.get_main_thread() is None

    def test_completed_main_thread_still_accessible(
        self, thread_manager: ThreadManager
    ) -> None:
        thread = thread_manager.create_thread(
            {"role": "user", "content": "hello"}, is_main=True
        )
        thread.status = ThreadStatus.COMPLETED
        thread.completed_at = datetime.now(UTC)
        # Not cleaned up yet — should still be accessible
        assert thread_manager.get_main_thread() is thread


# ── Inject Queue ────────────────────────────────────────────────────────────


class TestInjectQueue:
    def test_inject_queue_exists(self, thread_manager: ThreadManager) -> None:
        thread = thread_manager.create_thread({"role": "user", "content": "hello"})
        assert thread.inject_queue is not None
        assert thread.inject_queue.empty()

    async def test_inject_queue_put_get(self, thread_manager: ThreadManager) -> None:
        thread = thread_manager.create_thread({"role": "user", "content": "hello"})
        msg = {"role": "user", "content": "injected"}
        thread.inject_queue.put_nowait(msg)
        assert not thread.inject_queue.empty()
        got = thread.inject_queue.get_nowait()
        assert got == msg
        assert thread.inject_queue.empty()
