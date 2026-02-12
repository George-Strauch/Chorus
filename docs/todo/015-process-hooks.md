# TODO 015 — Process Hooks (Backlog)

> **Status:** PENDING

## Objective

Enable tracked processes (from TODO 014) to spawn agent execution threads when specific events occur. A process hook is a trigger + instructions + context bundle: when the trigger fires, a new execution thread is created with the hook's instructions and context injected into its system prompt. This turns agents from reactive (respond to user messages) into event-driven (respond to program events).

## Use Cases

- A database migration finishes → agent thread spawns to verify the migration and run tests.
- A build fails → agent thread spawns to read the error log, diagnose, and attempt a fix.
- A long-running server receives a specific log line → agent thread spawns to handle it.
- A training run hits a checkpoint → agent thread spawns to evaluate metrics and decide whether to continue.
- A monitoring process detects a threshold breach → agent thread spawns to investigate.

## Hook Types

```python
@dataclass
class ProcessHook:
    trigger: HookTrigger          # on_exit, on_output_match, on_timeout
    instructions: str             # What to tell the LLM when the hook fires
    context: dict                 # State to inject (why this process exists, what to do)
    max_spawns: int = 1           # How many times this hook can fire (prevent infinite loops)
    spawned_count: int = 0        # How many times it has fired

class HookTrigger:
    type: str                     # "on_exit", "on_output_match", "on_timeout"
    # on_exit fields:
    exit_codes: list[int] | None  # None = any exit code, [0] = success only, [1] = failure only
    # on_output_match fields:
    pattern: str | None           # Regex matched against each output line
    # on_timeout fields:
    timeout_seconds: int | None   # Fire if process runs longer than this
```

## Context Propagation

The key insight: a hook-spawned thread needs to understand WHY the process exists and WHAT to do when the hook fires. This context is attached when the process is detached, not when the hook fires.

Example context:
```python
{
    "purpose": "Running database migration for auth module refactor",
    "on_success": "Run the test suite for auth. If tests pass, commit and push.",
    "on_failure": "Read the migration error log, diagnose the issue, and attempt to fix the migration file.",
    "related_files": ["migrations/0042_auth_refactor.py", "tests/test_auth.py"],
    "original_thread_summary": "User asked to refactor auth to use JWT tokens. Migration was generated and is now running.",
}
```

When the hook fires, the spawned thread receives a system prompt section:
```
A process hook has fired. Here is the context:

Process: `python manage.py migrate` (PID 48231)
Trigger: on_exit (exit code: 1 — failure)
Runtime: 2h 13m

Hook context:
  Purpose: Running database migration for auth module refactor
  Instructions: Read the migration error log, diagnose the issue, and attempt to fix the migration file.
  Related files: migrations/0042_auth_refactor.py, tests/test_auth.py
  Original thread summary: User asked to refactor auth to use JWT tokens...

Process output (last 50 lines):
  [stderr tail here]
```

## Safety

- **Recursion depth limit:** A hook-spawned thread can detach processes with hooks, creating recursion. Enforce a maximum depth (default 3). Track depth in the context dict.
- **Max spawns per hook:** Default 1. An `on_output_match` hook on a noisy process could fire thousands of times without this.
- **Rate limiting:** Hook-spawned threads go through the same channel rate limiter as user-initiated threads.
- **Permission inheritance:** Hook-spawned threads inherit the permission profile of the agent, not elevated permissions. They still go through ASK prompts for restricted actions.

## Edge Cases

- **Hook fires while bot is down:** On restart recovery (TODO 014), check for processes that exited while the bot was offline. Fire their `on_exit` hooks with the recovered exit code.
- **Multiple hooks on same event:** Execute sequentially, each spawning its own thread.
- **Hook fires during context full:** Hook-spawned threads get minimal context (hook context + process tail), not the full agent history. They are independent threads.
- **Process produces no output:** `on_output_match` hooks never fire. This is fine — the `on_exit` or `on_timeout` hooks cover this case.

## Dependencies

- **006-execution-threads**: Hook-spawned threads are execution threads.
- **014-process-management**: Hooks are attached to tracked processes.
