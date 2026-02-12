# TODO 014 — Long-Running Process Management (Backlog)

> **Status:** PENDING

## Objective

Implement tracking and management of long-running OS processes spawned by agents. When an agent starts a process that will run for an extended period (database migrations, build jobs, training runs, servers), Chorus tracks the PID, captures output to log files, and provides commands to inspect and kill them. This is distinct from execution threads (TODO 006) — a process is an OS subprocess, while a thread is an LLM conversation loop.

## Acceptance Criteria

- When an agent spawns a long-running process via bash, it can be "detached" and tracked independently of the execution thread that started it.
- Each tracked process stores: PID, command, working directory, start time, agent name, spawning thread ID, stdout/stderr log file paths, hook configuration, and attached context.
- `/process list [agent]` shows all tracked processes with PID, command, runtime, and status.
- `/process logs <pid> [lines]` shows the last N lines (default 50) of a process's combined output.
- `/process kill <pid>` sends SIGTERM, waits grace period, then SIGKILL.
- `/process kill all [agent]` kills all tracked processes for an agent.
- Output is written to log files on disk (not buffered in memory) — processes running for days could produce gigabytes.
- A rolling tail of the last N lines (configurable, default 100) is kept in memory for fast `/process logs` and context injection.
- On bot startup, the process table is scanned: PIDs that are still alive are reconnected, dead processes are marked as exited and their `on_exit` hooks are fired.
- Process status is included in the agent's context preamble alongside thread status.

## Process State

```python
@dataclass
class TrackedProcess:
    pid: int
    command: str
    working_directory: Path
    agent_name: str
    started_at: datetime
    spawned_by_thread: int | None
    stdout_log: Path
    stderr_log: Path
    status: ProcessStatus  # running, exited, killed, lost
    exit_code: int | None
    hooks: list[ProcessHook]
    context: dict  # Arbitrary context attached by the spawning thread
    rolling_tail: deque[str]  # Last N lines for quick access
```

## Context Injection

Each LLM call includes a summary of running processes:
```
Running processes:
  PID 48231: `python manage.py migrate` — running 2h13m, last line: "Migrating app_users 0042..."
  PID 48450: `npm run build -- --watch` — running 15m, last line: "Compiled successfully"
```

## Bot Restart Recovery

On startup:
1. Load all tracked processes from SQLite.
2. For each, check if PID is still alive: `os.kill(pid, 0)`.
3. If alive: reconnect to log files, resume tail monitoring.
4. If dead: mark as exited, attempt to read exit code from `/proc/<pid>/status` or mark as `lost`, fire `on_exit` hooks for processes that died while bot was down.

## Implementation Notes

- **Storage:** Process records in SQLite (new `processes` table). Log files in agent's directory under `processes/<pid>/`.
- **Output capture:** Use `asyncio.create_subprocess_shell` with stdout/stderr piped to files. A background `asyncio.Task` tails the files and updates the rolling buffer.
- **Detach mechanism:** The execution thread calls a `detach_process(proc, hooks, context)` function that registers the process with the `ProcessManager` and returns immediately, freeing the thread to continue other work.

## Dependencies

- **005-bash-execution**: Process spawning mechanism.
- **006-execution-threads**: Thread ID tracking for spawning thread.
- **Storage schema**: New `processes` table in SQLite.
