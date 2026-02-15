"""Tests for OutputMonitor — async line reading, logging, rolling tail."""

from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path

import pytest


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    d = tmp_path / "logs"
    d.mkdir()
    return d


@pytest.mark.asyncio
async def test_monitor_reads_stdout(log_dir: Path) -> None:
    """OutputMonitor reads stdout line-by-line and populates rolling tail."""
    from chorus.process.monitor import OutputMonitor

    process = await asyncio.create_subprocess_exec(
        "echo", "-e", "line1\nline2\nline3",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    tail: deque[str] = deque(maxlen=100)
    monitor = OutputMonitor(
        pid=process.pid or 0,
        process=process,
        log_dir=log_dir,
        rolling_tail=tail,
    )
    monitor.start()
    # Wait for process to finish and monitor to complete
    await asyncio.wait_for(process.wait(), timeout=5.0)
    await asyncio.sleep(0.1)  # Let monitor finish processing
    await monitor.stop()

    # Check rolling tail has lines
    assert len(tail) > 0
    tail_list = list(tail)
    assert "line1" in tail_list[0]

    # Check log file was written
    stdout_log = log_dir / "stdout.log"
    assert stdout_log.exists()
    content = stdout_log.read_text()
    assert "line1" in content


@pytest.mark.asyncio
async def test_monitor_reads_stderr(log_dir: Path) -> None:
    """OutputMonitor reads stderr and prefixes with 'err: '."""
    from chorus.process.monitor import OutputMonitor

    process = await asyncio.create_subprocess_shell(
        "echo 'error line' >&2",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    tail: deque[str] = deque(maxlen=100)
    monitor = OutputMonitor(
        pid=process.pid or 0,
        process=process,
        log_dir=log_dir,
        rolling_tail=tail,
    )
    monitor.start()
    await asyncio.wait_for(process.wait(), timeout=5.0)
    await asyncio.sleep(0.1)
    await monitor.stop()

    tail_list = list(tail)
    assert any("err: " in line for line in tail_list)


@pytest.mark.asyncio
async def test_monitor_on_line_callback(log_dir: Path) -> None:
    """OutputMonitor fires on_line callback for each line."""
    from chorus.process.monitor import OutputMonitor

    lines_received: list[tuple[int, str, str]] = []

    async def on_line(pid: int, stream: str, line: str) -> None:
        lines_received.append((pid, stream, line))

    process = await asyncio.create_subprocess_exec(
        "echo", "hello",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    pid = process.pid or 0
    tail: deque[str] = deque(maxlen=100)
    monitor = OutputMonitor(
        pid=pid,
        process=process,
        log_dir=log_dir,
        rolling_tail=tail,
        on_line=on_line,
    )
    monitor.start()
    await asyncio.wait_for(process.wait(), timeout=5.0)
    await asyncio.sleep(0.1)
    await monitor.stop()

    assert len(lines_received) > 0
    assert lines_received[0][0] == pid
    assert lines_received[0][1] == "stdout"
    assert "hello" in lines_received[0][2]


@pytest.mark.asyncio
async def test_monitor_on_exit_callback(log_dir: Path) -> None:
    """OutputMonitor fires on_exit callback when process exits."""
    from chorus.process.monitor import OutputMonitor

    exit_events: list[tuple[int, int | None]] = []

    async def on_exit(pid: int, code: int | None) -> None:
        exit_events.append((pid, code))

    process = await asyncio.create_subprocess_exec(
        "true",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    pid = process.pid or 0
    tail: deque[str] = deque(maxlen=100)
    monitor = OutputMonitor(
        pid=pid,
        process=process,
        log_dir=log_dir,
        rolling_tail=tail,
        on_exit=on_exit,
    )
    monitor.start()
    await asyncio.wait_for(process.wait(), timeout=5.0)
    await asyncio.sleep(0.1)
    await monitor.stop()

    assert len(exit_events) == 1
    assert exit_events[0][0] == pid
    assert exit_events[0][1] == 0  # exit code 0


@pytest.mark.asyncio
async def test_monitor_nonzero_exit(log_dir: Path) -> None:
    """on_exit reports non-zero exit codes."""
    from chorus.process.monitor import OutputMonitor

    exit_events: list[tuple[int, int | None]] = []

    async def on_exit(pid: int, code: int | None) -> None:
        exit_events.append((pid, code))

    process = await asyncio.create_subprocess_exec(
        "false",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    pid = process.pid or 0
    tail: deque[str] = deque(maxlen=100)
    monitor = OutputMonitor(
        pid=pid,
        process=process,
        log_dir=log_dir,
        rolling_tail=tail,
        on_exit=on_exit,
    )
    monitor.start()
    await asyncio.wait_for(process.wait(), timeout=5.0)
    await asyncio.sleep(0.1)
    await monitor.stop()

    assert len(exit_events) == 1
    assert exit_events[0][1] == 1  # false exits with 1


@pytest.mark.asyncio
async def test_monitor_rolling_tail_maxlen(log_dir: Path) -> None:
    """Rolling tail respects its maxlen."""
    from chorus.process.monitor import OutputMonitor

    # Generate more than 100 lines
    cmd = "for i in $(seq 1 150); do echo line_$i; done"
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    tail: deque[str] = deque(maxlen=100)
    monitor = OutputMonitor(
        pid=process.pid or 0,
        process=process,
        log_dir=log_dir,
        rolling_tail=tail,
    )
    monitor.start()
    await asyncio.wait_for(process.wait(), timeout=5.0)
    await asyncio.sleep(0.2)
    await monitor.stop()

    assert len(tail) == 100
    # Should have the last 100 lines (51-150)
    assert "line_51" in list(tail)[0]


@pytest.mark.asyncio
async def test_monitor_stop_cancels(log_dir: Path) -> None:
    """stop() cancels the monitor even if the process is still running."""
    from chorus.process.monitor import OutputMonitor

    process = await asyncio.create_subprocess_exec(
        "sleep", "60",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    tail: deque[str] = deque(maxlen=100)
    monitor = OutputMonitor(
        pid=process.pid or 0,
        process=process,
        log_dir=log_dir,
        rolling_tail=tail,
    )
    monitor.start()
    await asyncio.sleep(0.1)
    await monitor.stop()  # Should not hang
    process.kill()
    await process.wait()


@pytest.mark.asyncio
async def test_monitor_creates_log_dir(tmp_path: Path) -> None:
    """OutputMonitor creates the log directory if it doesn't exist."""
    from chorus.process.monitor import OutputMonitor

    nested = tmp_path / "deep" / "nested" / "dir"
    process = await asyncio.create_subprocess_exec(
        "echo", "test",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    tail: deque[str] = deque(maxlen=100)
    monitor = OutputMonitor(
        pid=process.pid or 0,
        process=process,
        log_dir=nested,
        rolling_tail=tail,
    )
    monitor.start()
    await asyncio.wait_for(process.wait(), timeout=5.0)
    await asyncio.sleep(0.1)
    await monitor.stop()

    assert nested.exists()
