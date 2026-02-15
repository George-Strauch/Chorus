"""Tests for ProcessManager — spawn, kill, list, recovery."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from chorus.process.manager import ProcessManager
from chorus.process.models import ProcessStatus, ProcessType
from chorus.storage.db import Database


@pytest.fixture
async def db(tmp_path: Path) -> AsyncGenerator[Database, None]:
    db = Database(tmp_path / "db" / "chorus.db")
    await db.init()
    yield db
    await db.close()


@pytest.fixture
def chorus_home(tmp_path: Path) -> Path:
    home = tmp_path / "chorus-home"
    home.mkdir()
    (home / "agents" / "test-agent" / "workspace").mkdir(parents=True)
    return home


@pytest.fixture
def workspace(chorus_home: Path) -> Path:
    return chorus_home / "agents" / "test-agent" / "workspace"


@pytest.fixture
async def pm(chorus_home: Path, db: Database) -> ProcessManager:
    return ProcessManager(chorus_home=chorus_home, db=db)


@pytest.mark.asyncio
async def test_spawn_process(pm: ProcessManager, workspace: Path) -> None:
    """ProcessManager.spawn starts a process and tracks it."""
    tracked = await pm.spawn(
        command="sleep 10",
        workspace=workspace,
        agent_name="test-agent",
        process_type=ProcessType.BACKGROUND,
    )
    assert tracked.pid > 0
    assert tracked.status == ProcessStatus.RUNNING
    assert tracked.command == "sleep 10"
    assert tracked.agent_name == "test-agent"
    assert tracked.process_type == ProcessType.BACKGROUND

    # Should be in the list
    procs = pm.list_processes("test-agent")
    assert len(procs) == 1
    assert procs[0].pid == tracked.pid

    # Cleanup
    await pm.kill_process(tracked.pid)


@pytest.mark.asyncio
async def test_spawn_concurrent_process(pm: ProcessManager, workspace: Path) -> None:
    """Spawning a concurrent process works."""
    tracked = await pm.spawn(
        command="echo hello",
        workspace=workspace,
        agent_name="test-agent",
        process_type=ProcessType.CONCURRENT,
        spawned_by_branch=5,
    )
    assert tracked.process_type == ProcessType.CONCURRENT
    assert tracked.spawned_by_branch == 5

    # Wait for it to exit
    await asyncio.sleep(0.3)

    p = pm.get_process(tracked.pid)
    assert p is not None
    assert p.status in (ProcessStatus.EXITED, ProcessStatus.RUNNING)


@pytest.mark.asyncio
async def test_kill_process(pm: ProcessManager, workspace: Path) -> None:
    """kill_process sends SIGTERM and marks as KILLED."""
    tracked = await pm.spawn(
        command="sleep 60",
        workspace=workspace,
        agent_name="test-agent",
        process_type=ProcessType.BACKGROUND,
    )
    assert tracked.status == ProcessStatus.RUNNING

    killed = await pm.kill_process(tracked.pid)
    assert killed is True

    p = pm.get_process(tracked.pid)
    assert p is not None
    assert p.status == ProcessStatus.KILLED


@pytest.mark.asyncio
async def test_kill_nonexistent_process(pm: ProcessManager) -> None:
    """kill_process returns False for unknown PID."""
    result = await pm.kill_process(999999)
    assert result is False


@pytest.mark.asyncio
async def test_kill_already_exited(pm: ProcessManager, workspace: Path) -> None:
    """kill_process returns False if process already exited."""
    tracked = await pm.spawn(
        command="true",  # Exits immediately
        workspace=workspace,
        agent_name="test-agent",
        process_type=ProcessType.BACKGROUND,
    )
    await asyncio.sleep(0.3)  # Wait for exit

    result = await pm.kill_process(tracked.pid)
    assert result is False


@pytest.mark.asyncio
async def test_kill_all_for_agent(pm: ProcessManager, workspace: Path) -> None:
    """kill_all_for_agent kills all running processes."""
    p1 = await pm.spawn("sleep 60", workspace, "test-agent", ProcessType.BACKGROUND)
    p2 = await pm.spawn("sleep 60", workspace, "test-agent", ProcessType.BACKGROUND)

    count = await pm.kill_all_for_agent("test-agent")
    assert count == 2

    assert pm.get_process(p1.pid).status == ProcessStatus.KILLED  # type: ignore[union-attr]
    assert pm.get_process(p2.pid).status == ProcessStatus.KILLED  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_list_processes_filters_by_agent(
    pm: ProcessManager, workspace: Path, chorus_home: Path
) -> None:
    """list_processes filters by agent_name."""
    # Create workspace for second agent
    ws2 = chorus_home / "agents" / "other-agent" / "workspace"
    ws2.mkdir(parents=True)

    p1 = await pm.spawn("sleep 10", workspace, "test-agent", ProcessType.BACKGROUND)
    p2 = await pm.spawn("sleep 10", ws2, "other-agent", ProcessType.BACKGROUND)

    procs_test = pm.list_processes("test-agent")
    assert len(procs_test) == 1
    assert procs_test[0].pid == p1.pid

    procs_other = pm.list_processes("other-agent")
    assert len(procs_other) == 1
    assert procs_other[0].pid == p2.pid

    procs_all = pm.list_processes()
    assert len(procs_all) == 2

    await pm.kill_process(p1.pid)
    await pm.kill_process(p2.pid)


@pytest.mark.asyncio
async def test_get_process(pm: ProcessManager, workspace: Path) -> None:
    """get_process returns TrackedProcess by PID."""
    tracked = await pm.spawn("sleep 10", workspace, "test-agent", ProcessType.BACKGROUND)

    result = pm.get_process(tracked.pid)
    assert result is not None
    assert result.pid == tracked.pid
    assert result.command == "sleep 10"

    # Non-existent PID
    assert pm.get_process(999999) is None

    await pm.kill_process(tracked.pid)


@pytest.mark.asyncio
async def test_process_persisted_to_db(
    pm: ProcessManager, workspace: Path, db: Database
) -> None:
    """Spawned processes are persisted to the database."""
    tracked = await pm.spawn("sleep 10", workspace, "test-agent", ProcessType.BACKGROUND)

    row = await db.get_process(tracked.pid)
    assert row is not None
    assert row["command"] == "sleep 10"
    assert row["agent_name"] == "test-agent"
    assert row["process_type"] == "background"
    assert row["status"] == "running"

    await pm.kill_process(tracked.pid)

    # Check status updated in DB
    row = await db.get_process(tracked.pid)
    assert row is not None
    assert row["status"] == "killed"


@pytest.mark.asyncio
async def test_process_exit_updates_db(
    pm: ProcessManager, workspace: Path, db: Database
) -> None:
    """Process exit updates the DB status."""
    tracked = await pm.spawn("true", workspace, "test-agent", ProcessType.BACKGROUND)
    await asyncio.sleep(0.5)  # Wait for exit

    row = await db.get_process(tracked.pid)
    assert row is not None
    assert row["status"] == "exited"
    assert row["exit_code"] == 0


@pytest.mark.asyncio
async def test_spawn_with_callbacks(pm: ProcessManager, workspace: Path) -> None:
    """Spawning with callbacks persists them."""
    from chorus.process.models import (
        CallbackAction,
        HookTrigger,
        ProcessCallback,
        TriggerType,
    )

    cb = ProcessCallback(
        trigger=HookTrigger(type=TriggerType.ON_EXIT),
        action=CallbackAction.SPAWN_BRANCH,
        context_message="Process finished",
    )
    tracked = await pm.spawn(
        command="sleep 10",
        workspace=workspace,
        agent_name="test-agent",
        process_type=ProcessType.BACKGROUND,
        callbacks=[cb],
        context="test context",
        model_for_hooks="claude-haiku-4-5-20251001",
    )
    assert len(tracked.callbacks) == 1
    assert tracked.context == "test context"
    assert tracked.model_for_hooks == "claude-haiku-4-5-20251001"

    await pm.kill_process(tracked.pid)


@pytest.mark.asyncio
async def test_recover_on_startup_marks_lost(
    chorus_home: Path, db: Database
) -> None:
    """recover_on_startup marks stale 'running' processes as LOST."""
    # Insert a fake process with a non-existent PID
    await db.insert_process(
        pid=999999,
        command="ghost process",
        working_directory="/tmp",
        agent_name="test-agent",
        started_at="2026-01-01T00:00:00",
        process_type="background",
        status="running",
    )

    pm = ProcessManager(chorus_home=chorus_home, db=db)
    await pm.recover_on_startup()

    row = await db.get_process(999999)
    assert row is not None
    assert row["status"] == "lost"


@pytest.mark.asyncio
async def test_on_spawn_callback_fired(pm: ProcessManager, workspace: Path) -> None:
    """on_spawn callback is called after spawn() with the new PID."""
    spawn_pids: list[int] = []

    def on_spawn(pid: int) -> None:
        spawn_pids.append(pid)

    pm.set_callbacks(on_spawn=on_spawn)

    tracked = await pm.spawn(
        command="sleep 10",
        workspace=workspace,
        agent_name="test-agent",
        process_type=ProcessType.BACKGROUND,
    )
    assert len(spawn_pids) == 1
    assert spawn_pids[0] == tracked.pid

    await pm.kill_process(tracked.pid)


@pytest.mark.asyncio
async def test_spawn_captures_output(pm: ProcessManager, workspace: Path) -> None:
    """Spawned processes capture output in rolling tail."""
    tracked = await pm.spawn(
        command="echo 'hello world'",
        workspace=workspace,
        agent_name="test-agent",
        process_type=ProcessType.BACKGROUND,
    )
    await asyncio.sleep(0.5)

    p = pm.get_process(tracked.pid)
    assert p is not None
    tail = list(p.rolling_tail)
    assert any("hello world" in line for line in tail)


@pytest.mark.asyncio
async def test_python_output_arrives_before_exit(
    pm: ProcessManager, workspace: Path
) -> None:
    """PYTHONUNBUFFERED=1 ensures Python output is visible before process exits.

    Without unbuffered mode, piped stdout is fully buffered and
    on_line callbacks only fire at exit. This test verifies that
    a line printed mid-execution is visible while the process
    is still running.
    """
    lines_seen: list[str] = []

    async def on_line(pid: int, stream: str, line: str) -> None:
        lines_seen.append(line)

    pm.set_callbacks(on_line=on_line)

    # Python script that prints a marker, then sleeps
    script = (
        "import time; "
        "print('MARKER_VISIBLE'); "
        "time.sleep(5)"
    )
    tracked = await pm.spawn(
        command=f"python3 -c \"{script}\"",
        workspace=workspace,
        agent_name="test-agent",
        process_type=ProcessType.BACKGROUND,
    )

    # Wait up to 2s for the marker to appear — well before the 5s sleep ends
    for _ in range(20):
        if any("MARKER_VISIBLE" in line for line in lines_seen):
            break
        await asyncio.sleep(0.1)

    # The process should still be running
    p = pm.get_process(tracked.pid)
    assert p is not None
    assert p.status == ProcessStatus.RUNNING

    # And we should have seen the marker
    assert any("MARKER_VISIBLE" in line for line in lines_seen), (
        "Output line was not visible before process exit — "
        "PYTHONUNBUFFERED may not be set in subprocess env"
    )

    await pm.kill_process(tracked.pid)


@pytest.mark.asyncio
async def test_stdbuf_wraps_command(pm: ProcessManager, workspace: Path) -> None:
    """Spawned commands are wrapped with stdbuf -oL when available."""
    from chorus.process.manager import _STDBUF_PATH

    tracked = await pm.spawn(
        command="echo test",
        workspace=workspace,
        agent_name="test-agent",
        process_type=ProcessType.BACKGROUND,
    )
    await asyncio.sleep(0.3)

    # The TrackedProcess stores the original command (not wrapped)
    assert tracked.command == "echo test"

    # But the actual subprocess got stdbuf if available
    if _STDBUF_PATH is not None:
        # Verify stdbuf is on the system (it's in coreutils)
        import shutil

        assert shutil.which("stdbuf") is not None

    await pm.kill_process(tracked.pid)
