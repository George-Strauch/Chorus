# TODO 032: Process Runner Tools & Discord UI

## Objective

Create `run_concurrent` and `run_background` tool handlers that agents use to start long-running processes, a ProcessStatusEmbed for background process Discord feedback, register the tools, and extend ToolExecutionContext for process-related fields.

## Dependencies

- TODO 030 (process management core)
- TODO 031 (hooks and callback builder)

## Components

### Tool Handlers

**run_concurrent(command, instructions):**
1. Permission check (`tool:run_concurrent:<command>`)
2. Reuse `_check_blocklist()` from bash.py
3. Build callbacks from instructions
4. Spawn via ProcessManager with type=CONCURRENT
5. Return immediately with `{"pid": ..., "status": "running"}`

**run_background(command, instructions, model=None):**
1. Permission check (`tool:run_background:<command>`)
2. Reuse `_check_blocklist()` from bash.py
3. Build callbacks from instructions
4. Spawn via ProcessManager with type=BACKGROUND
5. Return immediately with `{"pid": ..., "status": "running"}`

### ProcessStatusEmbed

Live-updating Discord embed for background processes:
- Color-coded: green=running, red=error exit, gray=clean exit
- Fields: command, PID, uptime, status, last N lines of output
- Updates periodically and on events

### Integration

- Register tools in registry.py
- Add category mappings in tool_loop.py
- Add ASK patterns in engine.py standard preset
- Extend ToolExecutionContext with process_manager, thread_manager, branch_id, inject_queue, channel
- Pass new fields in bot.py _make_llm_runner

## Tests

- `tests/test_tools/test_run_process.py`
- `tests/test_ui/test_process_embed.py`
