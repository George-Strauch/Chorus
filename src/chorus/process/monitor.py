"""OutputMonitor — async line-by-line reader for process stdout/stderr."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

logger = logging.getLogger("chorus.process.monitor")

# Type alias for the line callback used by hooks
LineCallback = Callable[[int, str, str], Coroutine[Any, Any, None]]
# (pid, stream_name, line_text) → awaitable

# Type alias for the exit callback
ExitCallback = Callable[[int, int | None], Coroutine[Any, Any, None]]
# (pid, exit_code) → awaitable


class OutputMonitor:
    """Monitors stdout/stderr of a subprocess, writing logs and maintaining a rolling tail.

    Parameters
    ----------
    pid:
        Process ID being monitored.
    process:
        The asyncio subprocess.
    log_dir:
        Directory for stdout.log / stderr.log files.
    rolling_tail:
        Shared deque for the most recent output lines.
    on_line:
        Async callback invoked for each output line (for hook evaluation).
    on_exit:
        Async callback invoked when the process exits.
    """

    def __init__(
        self,
        pid: int,
        process: asyncio.subprocess.Process,
        log_dir: Path,
        rolling_tail: deque[str],
        on_line: LineCallback | None = None,
        on_exit: ExitCallback | None = None,
    ) -> None:
        self._pid = pid
        self._process = process
        self._log_dir = log_dir
        self._rolling_tail = rolling_tail
        self._on_line = on_line
        self._on_exit = on_exit
        self._task: asyncio.Task[None] | None = None
        self._stdout_log: Path | None = None
        self._stderr_log: Path | None = None

    @property
    def stdout_log(self) -> str | None:
        return str(self._stdout_log) if self._stdout_log else None

    @property
    def stderr_log(self) -> str | None:
        return str(self._stderr_log) if self._stderr_log else None

    def start(self) -> None:
        """Start the monitoring task."""
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Cancel the monitoring task."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        """Main monitoring loop: read both streams, then wait for exit."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._stdout_log = self._log_dir / "stdout.log"
        self._stderr_log = self._log_dir / "stderr.log"

        try:
            # Read stdout and stderr concurrently
            tasks: list[asyncio.Task[None]] = []
            if self._process.stdout is not None:
                tasks.append(
                    asyncio.create_task(
                        self._read_stream(self._process.stdout, "stdout", self._stdout_log)
                    )
                )
            if self._process.stderr is not None:
                tasks.append(
                    asyncio.create_task(
                        self._read_stream(self._process.stderr, "stderr", self._stderr_log)
                    )
                )

            if tasks:
                await asyncio.gather(*tasks)

            # Wait for process exit
            exit_code = await self._process.wait()

            # Fire exit callback
            if self._on_exit is not None:
                try:
                    await self._on_exit(self._pid, exit_code)
                except Exception:
                    logger.warning("on_exit callback error for pid %d", self._pid, exc_info=True)

        except asyncio.CancelledError:
            logger.debug("OutputMonitor for pid %d cancelled", self._pid)
            raise
        except Exception:
            logger.exception("OutputMonitor error for pid %d", self._pid)

    async def _read_stream(
        self,
        stream: asyncio.StreamReader,
        stream_name: str,
        log_path: Path,
    ) -> None:
        """Read lines from a stream, write to log, update rolling tail."""
        with log_path.open("w", encoding="utf-8") as f:
            while True:
                try:
                    line_bytes = await stream.readline()
                except Exception:
                    break
                if not line_bytes:
                    break

                line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                f.write(line + "\n")
                f.flush()

                # Update rolling tail
                prefix = "err: " if stream_name == "stderr" else ""
                self._rolling_tail.append(f"{prefix}{line}")

                # Fire line callback
                if self._on_line is not None:
                    try:
                        await self._on_line(self._pid, stream_name, line)
                    except Exception:
                        logger.warning(
                            "on_line callback error for pid %d", self._pid, exc_info=True
                        )
