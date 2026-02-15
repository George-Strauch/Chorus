# TODO 030: Process Management Core

## Objective

Build the core infrastructure for managing long-running processes spawned by agents. This includes data models, a ProcessManager singleton, an OutputMonitor for async stdout/stderr reading, a DB table for process persistence, and `/process` slash commands.

## Dependencies

- TODO 005 (bash execution — reuses `_check_blocklist`, `_sanitized_env`)
- TODO 006 (execution threads — OutputMonitor interacts with ThreadManager)
- TODO 007 (context management — process status injected into LLM context)

## Data Models

### Enums

- `ProcessStatus`: RUNNING, EXITED, KILLED, LOST
- `ProcessType`: CONCURRENT, BACKGROUND
- `TriggerType`: ON_EXIT, ON_OUTPUT_MATCH, ON_TIMEOUT
- `ExitFilter`: ANY, SUCCESS, FAILURE
- `CallbackAction`: STOP_PROCESS, STOP_BRANCH, INJECT_CONTEXT, SPAWN_BRANCH

### Dataclasses

- `HookTrigger(type, exit_filter, pattern, timeout_seconds)`
- `ProcessCallback(trigger, action, context_message, output_delay_seconds, max_fires, fire_count)`
- `TrackedProcess(pid, command, working_directory, agent_name, started_at, process_type, spawned_by_branch, stdout_log, stderr_log, status, exit_code, callbacks, context, rolling_tail, model_for_hooks, hook_recursion_depth, discord_message_id)`

## Components

### ProcessManager

Central singleton owning all process lifecycle.

- `spawn(command, workspace, agent_name, process_type, callbacks, ...)` → TrackedProcess
- `kill_process(pid)` — SIGTERM → grace → SIGKILL
- `kill_all_for_agent(agent_name)`
- `list_processes(agent_name=None)`
- `get_process(pid)` → TrackedProcess | None
- `recover_on_startup()` — load from DB, check PIDs, reconnect or mark dead

### OutputMonitor

Async task per process.

- Reads stdout/stderr via `asyncio.StreamReader` line-by-line
- Writes to log files (`~/.chorus-agents/agents/<name>/processes/<pid>/stdout.log`)
- Maintains `deque(maxlen=100)` rolling tail
- Evaluates regex triggers on each line (delegated to hooks in TODO 031)

### DB Schema

```sql
CREATE TABLE IF NOT EXISTS processes (
    pid INTEGER PRIMARY KEY,
    command TEXT NOT NULL,
    working_directory TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    process_type TEXT NOT NULL,
    spawned_by_branch INTEGER,
    stdout_log TEXT,
    stderr_log TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    exit_code INTEGER,
    callbacks_json TEXT,
    context_json TEXT,
    model_for_hooks TEXT,
    hook_recursion_depth INTEGER DEFAULT 0,
    discord_message_id INTEGER
);
```

### Process Commands

- `/process list` — list running/recent processes for the agent
- `/process kill <pid>` — kill a specific process
- `/process logs <pid>` — show last N lines of stdout/stderr

## Acceptance Criteria

1. ProcessManager can spawn processes, track them, and kill them
2. OutputMonitor reads output line-by-line and maintains rolling tail
3. Processes persist to SQLite and survive bot restarts (recovery on startup)
4. Process status injected into LLM context preamble
5. `/process list|kill|logs` commands work
6. All existing tests continue to pass

## Tests

- `tests/test_process/test_models.py` — enum/dataclass construction and serialization
- `tests/test_process/test_manager.py` — spawn, kill, list, recovery
- `tests/test_process/test_monitor.py` — line reading, log writing, rolling tail
- `tests/test_commands/test_process_commands.py` — slash command behavior
- DB additions in existing `test_db.py`
