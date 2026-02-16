"""Sandboxed bash execution within agent workspaces.

Commands run as subprocesses with ``cwd`` set to the agent's workspace,
sanitized environment, timeout handling, output truncation, and a
best-effort command blocklist.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from chorus.permissions.engine import PermissionResult, check, format_action

if TYPE_CHECKING:
    from pathlib import Path

    from chorus.permissions.engine import PermissionProfile

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CommandDeniedError(Exception):
    """Raised when the permission engine denies a command."""


class CommandNeedsApprovalError(Exception):
    """Raised when the permission engine returns ASK for a command."""

    def __init__(self, command: str, message: str = "") -> None:
        self.command = command
        super().__init__(message or f"Command needs approval: {command}")


class CommandBlockedError(Exception):
    """Raised when a command matches the safety blocklist."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BashResult:
    """Structured result from a bash execution."""

    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: int
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "duration_ms": self.duration_ms,
            "truncated": self.truncated,
        }


# ---------------------------------------------------------------------------
# Environment sanitization
# ---------------------------------------------------------------------------

ALLOWED_ENV_VARS: frozenset[str] = frozenset(
    {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR", "SCOPE_PATH"}
)


def _sanitized_env(
    workspace: Path,
    overrides: dict[str, str] | None = None,
    host_execution: bool = False,
    scope_home: Path | None = None,
) -> dict[str, str]:
    """Build a sanitized environment for subprocess execution.

    When *host_execution* is False (default): only variables in
    :data:`ALLOWED_ENV_VARS` are carried over and ``HOME`` is jailed
    to the workspace directory.

    When *host_execution* is True: the full host environment is passed
    through and ``HOME`` is NOT jailed. This gives the subprocess access
    to the host's full development environment (compilers, tools, etc.).

    When *scope_home* is set, ``HOME`` is overridden to point there
    regardless of other settings. This allows git/ssh to find credentials
    at the host user's home directory (e.g. ``/mnt/host``).

    Always sets ``PYTHONUNBUFFERED=1`` so Python subprocesses flush
    stdout/stderr immediately rather than buffering until exit.
    """
    if host_execution:
        env = dict(os.environ)
    else:
        env = {k: v for k, v in os.environ.items() if k in ALLOWED_ENV_VARS}
        env["HOME"] = str(workspace)
    if scope_home is not None:
        env["HOME"] = str(scope_home)
    # Force Python subprocesses to flush stdout/stderr line-by-line.
    # Without this, piped stdout is fully buffered (~4-8KB) and output
    # only arrives when the buffer fills or the process exits — breaking
    # ON_OUTPUT_MATCH hooks that need to see lines in real time.
    env["PYTHONUNBUFFERED"] = "1"
    if overrides:
        env.update(overrides)
    return env


# ---------------------------------------------------------------------------
# Scope-path detection
# ---------------------------------------------------------------------------


def _targets_scope_path(command: str, workspace: Path, scope_path: Path | None) -> bool:
    """Return True if *command* or *workspace* references *scope_path*.

    Used to auto-detect when a bash command targets the host filesystem
    so we can enable full environment passthrough and set HOME accordingly.
    """
    if scope_path is None:
        return False
    sp = str(scope_path)
    # Workspace itself is under scope_path
    if str(workspace.resolve()).startswith(sp):
        return True
    # Command references scope_path literally or via env var
    if sp in command or "$SCOPE_PATH" in command or "${SCOPE_PATH}" in command:
        return True
    return False


# ---------------------------------------------------------------------------
# Command blocklist
# ---------------------------------------------------------------------------

BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"rm\s+-[^\s]*r[^\s]*f[^\s]*\s+/\s*$"),  # rm -rf /
    re.compile(r"rm\s+-[^\s]*f[^\s]*r[^\s]*\s+/\s*$"),  # rm -fr /
    re.compile(r":\(\)\s*\{.*\}"),  # fork bomb
    re.compile(r"dd\s+if=/dev/(zero|random)"),  # disk fill
    re.compile(r"mkfs"),  # format disk
    re.compile(r">\s*/dev/sd[a-z]"),  # overwrite disk
]


def _check_blocklist(command: str) -> None:
    """Raise :exc:`CommandBlockedError` if *command* matches a blocked pattern."""
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(command):
            raise CommandBlockedError(f"Command blocked by safety filter: {command!r}")


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


def _truncate_output(output: str, max_len: int) -> tuple[str, bool]:
    """Truncate *output* to *max_len* chars, keeping the tail.

    Returns ``(output, truncated)`` where *truncated* is True if
    the output was shortened.
    """
    if len(output) <= max_len:
        return output, False
    header = f"[Output truncated: showing last {max_len} chars of {len(output)} chars]\n"
    return header + output[-max_len:], True


# ---------------------------------------------------------------------------
# Process tracker
# ---------------------------------------------------------------------------


class ProcessTracker:
    """Track running subprocesses per agent for status reporting and cleanup."""

    def __init__(self) -> None:
        self._processes: dict[str, dict[int, asyncio.subprocess.Process]] = {}
        self._lock = asyncio.Lock()

    async def register(self, agent_name: str, process: asyncio.subprocess.Process) -> None:
        async with self._lock:
            if agent_name not in self._processes:
                self._processes[agent_name] = {}
            if process.pid is not None:
                self._processes[agent_name][process.pid] = process

    async def unregister(self, agent_name: str, process: asyncio.subprocess.Process) -> None:
        async with self._lock:
            agent_procs = self._processes.get(agent_name)
            if agent_procs and process.pid is not None:
                agent_procs.pop(process.pid, None)

    def get_running(self, agent_name: str) -> list[asyncio.subprocess.Process]:
        agent_procs = self._processes.get(agent_name, {})
        return [p for p in agent_procs.values() if p.returncode is None]

    async def kill_all(self, agent_name: str) -> int:
        """Kill all running processes for *agent_name*. Returns count killed."""
        async with self._lock:
            agent_procs = self._processes.get(agent_name, {})
            killed = 0
            for process in list(agent_procs.values()):
                if process.returncode is None:
                    process.kill()
                    killed += 1
            self._processes.pop(agent_name, None)
            return killed


_process_tracker: ProcessTracker | None = None


def get_process_tracker() -> ProcessTracker:
    """Return the module-level :class:`ProcessTracker` singleton."""
    global _process_tracker  # noqa: PLW0603
    if _process_tracker is None:
        _process_tracker = ProcessTracker()
    return _process_tracker


# ---------------------------------------------------------------------------
# Core execution
# ---------------------------------------------------------------------------

_DEFAULT_SIGTERM_GRACE_SECONDS = 5.0


async def bash_execute(
    command: str,
    workspace: Path,
    profile: PermissionProfile,
    timeout: float = 120.0,
    max_output_len: int = 50_000,
    env_overrides: dict[str, str] | None = None,
    agent_name: str = "",
    sigterm_grace: float = _DEFAULT_SIGTERM_GRACE_SECONDS,
    host_execution: bool = False,
    scope_path: Path | None = None,
) -> BashResult:
    """Execute *command* in a subprocess within *workspace*.

    Parameters
    ----------
    command:
        Shell command to execute.
    workspace:
        Agent workspace directory (used as ``cwd`` and ``HOME``).
    profile:
        Permission profile to check against.
    timeout:
        Maximum execution time in seconds.
    max_output_len:
        Maximum characters per output stream before truncation.
    env_overrides:
        Additional environment variables to set.
    agent_name:
        Agent identifier for process tracking.
    sigterm_grace:
        Seconds to wait after SIGTERM before escalating to SIGKILL.

    Raises
    ------
    CommandBlockedError
        If the command matches the safety blocklist.
    CommandDeniedError
        If the permission engine denies the command.
    CommandNeedsApprovalError
        If the permission engine returns ASK.
    """
    # 1. Blocklist check (unconditional)
    _check_blocklist(command)

    # 2. Permission check
    action = format_action("bash", command)
    result = check(action, profile)
    if result is PermissionResult.DENY:
        raise CommandDeniedError(f"Permission denied: {command!r}")
    if result is PermissionResult.ASK:
        raise CommandNeedsApprovalError(command)

    # 3. Build sanitized environment
    # When the command targets the host filesystem (scope_path), auto-enable
    # full env passthrough and set HOME=scope_path so git/ssh find credentials.
    effective_host_exec = host_execution
    effective_scope_home: Path | None = None
    if _targets_scope_path(command, workspace, scope_path):
        effective_host_exec = True
        effective_scope_home = scope_path
    env = _sanitized_env(
        workspace, env_overrides,
        host_execution=effective_host_exec,
        scope_home=effective_scope_home,
    )

    # 4. Execute
    tracker = get_process_tracker()
    start = time.monotonic()
    truncated = False

    process = await asyncio.create_subprocess_shell(
        command,
        cwd=workspace,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    await tracker.register(agent_name, process)
    try:
        try:
            raw_stdout, raw_stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            # Graceful shutdown: SIGTERM first
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            # Wait for grace period, then SIGKILL if still alive
            try:
                await asyncio.wait_for(
                    process.communicate(),
                    timeout=sigterm_grace,
                )
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                # Reap the process
                with contextlib.suppress(Exception):
                    await process.communicate()

            elapsed_ms = int((time.monotonic() - start) * 1000)
            return BashResult(
                command=command,
                exit_code=process.returncode,
                stdout="",
                stderr="",
                timed_out=True,
                duration_ms=elapsed_ms,
            )
    finally:
        await tracker.unregister(agent_name, process)

    elapsed_ms = int((time.monotonic() - start) * 1000)

    # 5. Decode output (binary-safe)
    stdout = raw_stdout.decode("utf-8", errors="replace")
    stderr = raw_stderr.decode("utf-8", errors="replace")

    # 6. Truncate if needed
    stdout, stdout_trunc = _truncate_output(stdout, max_output_len)
    stderr, stderr_trunc = _truncate_output(stderr, max_output_len)
    truncated = stdout_trunc or stderr_trunc

    return BashResult(
        command=command,
        exit_code=process.returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        duration_ms=elapsed_ms,
        truncated=truncated,
    )
