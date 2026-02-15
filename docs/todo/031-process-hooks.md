# TODO 031: Process Hooks & Callback System

## Objective

Build the hook dispatch system that fires callbacks when process events occur, and a callback builder that translates natural language instructions into structured callbacks using a cheap LLM call.

## Dependencies

- TODO 030 (process management core — ProcessManager, OutputMonitor, models)

## Components

### HookDispatcher

Evaluates callbacks against process events and executes actions.

**Trigger evaluation:**
- `ON_EXIT`: fires when process exits, filtered by ExitFilter (any/success/failure)
- `ON_OUTPUT_MATCH`: fires when a line matches the callback's regex pattern
- `ON_TIMEOUT`: fires after the specified timeout_seconds

**Actions:**
- `STOP_PROCESS` → `process_manager.kill_process(pid)`
- `STOP_BRANCH` → cancel the thread task via ThreadManager (concurrent only)
- `INJECT_CONTEXT` → push message into `ExecutionThread.inject_queue` (concurrent only)
- `SPAWN_BRANCH` → call `BranchSpawner.spawn_hook_branch()` with full context

**Output delay buffering:**
- On regex match, start `asyncio.sleep(delay)` timer
- During delay, output continues buffering in rolling tail
- After delay, callback fires with all buffered output since match
- Default delay from GlobalConfig.default_output_delay (2.0s)

**Safety:**
- Max recursion depth (default 3) — hook-spawned processes inherit depth + 1
- Max fires per callback (default 1 for on_output_match)
- Rate limiter: asyncio.Semaphore(3) on concurrent hook-spawned branches

### BranchSpawner Protocol

```python
class BranchSpawner(Protocol):
    async def spawn_hook_branch(
        self, agent_name: str, hook_context: str,
        model: str | None = None, recursion_depth: int = 0,
    ) -> None: ...
```

### CallbackBuilder

Inline Haiku call that translates NL instructions to structured callbacks.

- Agent provides NL `instructions` (e.g., "if it fails, read the logs and fix it")
- Cheap Haiku call translates to `list[ProcessCallback]`
- Fallback on failure: single `on_exit(any) → spawn_branch` callback

## Acceptance Criteria

1. HookDispatcher fires callbacks correctly for all trigger types
2. Output delay buffering works (waits, accumulates output, then fires)
3. Safety limits enforced (recursion depth, max fires, rate limiting)
4. CallbackBuilder translates NL to structured callbacks
5. HookDispatcher integrates with ProcessManager via set_callbacks()
6. All existing tests continue to pass

## Tests

- `tests/test_process/test_hooks.py` — trigger evaluation, action dispatch, delay buffering, safety
- `tests/test_process/test_callback_builder.py` — NL→callback translation, fallback
