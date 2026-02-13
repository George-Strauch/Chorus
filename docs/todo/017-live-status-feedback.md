# TODO 017 — Live Status Feedback

> **Status:** COMPLETED

## Objective

Implement a live status feedback system that gives visual feedback during agent execution without spamming messages. Two components: (1) a **live status embed** sent once per thread and edited in-place with real-time metrics (threads running, current step, token usage, elapsed time, tool calls), and (2) **bot presence management** that reflects aggregate processing state across all agents.

## Acceptance Criteria

### Live Status Embed
- When a thread starts executing, a single embed message is sent to the channel.
- The embed is edited in-place as execution progresses — no new messages per update.
- The embed displays:
  - Agent name and thread ID.
  - Current status (processing / waiting for permission / completed / errored).
  - Step count and current step description (e.g., "Step 3: Executing bash_execute").
  - Number of active threads for this agent.
  - Token usage (input + output, running total).
  - Number of LLM iterations completed.
  - Number of tool calls made (with list of tool names used).
  - Elapsed wall-clock time (live-updating).
- Edits are throttled to max 1 edit per 1.5 seconds to respect Discord rate limits. Intermediate events are batched.
- The final edit on completion shows a summary: total steps, total tokens, total tool calls, total time, and the tools used.
- If the thread errors or is killed, the embed reflects that with an error/cancelled status.

### Bot Presence
- The bot's Discord presence reflects aggregate execution state across all agents.
- When any thread is running: presence shows activity like `Processing N task(s) across M agent(s)`.
- When all threads are idle: presence shows `Idle | N agent(s) online` (or clears to default).
- Presence updates are debounced (max 1 update per 5 seconds) to respect Discord's presence rate limit.
- Multiple agents running concurrently is handled correctly — presence reflects the total, not just the last agent.

### Tool Loop Integration
- `run_tool_loop` accepts an optional `on_event` callback that fires at key lifecycle points.
- Events: `llm_call_start`, `llm_call_complete`, `tool_call_start`, `tool_call_complete`, `loop_complete`.
- Each event carries relevant data (tool name, usage delta, iteration count, etc.).
- The callback is non-blocking — status updates must not slow down execution. Errors in the callback are logged but do not interrupt the tool loop.
- The existing `ask_callback` is unaffected.

### General
- The status module is self-contained and optional — if no status callback is provided, execution proceeds as before.
- All status embed sends/edits are fire-and-forget with error handling (Discord failures don't crash the thread).
- Works correctly with concurrent threads on the same agent (each gets its own status embed).

## Tests to Write First

File: `tests/test_ui/__init__.py`
```
(empty)
```

File: `tests/test_ui/test_status.py`
```
# StatusEmbed building
test_initial_embed_has_agent_name_and_thread_id
test_initial_embed_shows_processing_status
test_embed_updates_step_count
test_embed_updates_token_usage
test_embed_updates_tool_call_count
test_embed_shows_tools_used_list
test_embed_shows_elapsed_time
test_embed_shows_active_thread_count
test_embed_shows_llm_iterations
test_final_embed_shows_completed_status
test_final_embed_shows_totals
test_error_embed_shows_error_status
test_cancelled_embed_shows_cancelled_status

# LiveStatusView lifecycle
test_live_status_sends_initial_embed_on_start
test_live_status_edits_embed_on_update
test_live_status_throttles_edits
test_live_status_batches_rapid_events
test_live_status_always_sends_final_update
test_live_status_handles_discord_send_failure
test_live_status_handles_discord_edit_failure
test_concurrent_threads_get_separate_embeds

# BotPresenceManager
test_presence_shows_processing_when_threads_active
test_presence_shows_idle_when_no_threads
test_presence_counts_multiple_agents
test_presence_debounces_rapid_updates
test_presence_handles_bot_not_ready
test_presence_updates_on_thread_start
test_presence_updates_on_thread_complete

# ToolLoopEvent integration
test_on_event_called_on_llm_call_start
test_on_event_called_on_llm_call_complete_with_usage
test_on_event_called_on_tool_call_start_with_tool_name
test_on_event_called_on_tool_call_complete
test_on_event_called_on_loop_complete
test_on_event_error_does_not_interrupt_loop
test_on_event_none_is_accepted_gracefully
test_on_event_receives_iteration_count
```

File: `tests/test_llm/test_tool_loop.py` (extend)
```
test_tool_loop_calls_on_event_callback
test_tool_loop_on_event_error_logged_not_raised
test_tool_loop_on_event_none_no_error
```

## Implementation Notes

1. **Module location:** `src/chorus/ui/__init__.py` and `src/chorus/ui/status.py`.

2. **ToolLoopEvent enum and data:**
   ```python
   class ToolLoopEventType(str, Enum):
       LLM_CALL_START = "llm_call_start"
       LLM_CALL_COMPLETE = "llm_call_complete"
       TOOL_CALL_START = "tool_call_start"
       TOOL_CALL_COMPLETE = "tool_call_complete"
       LOOP_COMPLETE = "loop_complete"

   @dataclass
   class ToolLoopEvent:
       type: ToolLoopEventType
       iteration: int
       tool_name: str | None = None
       usage_delta: Usage | None = None       # Token usage from this LLM call
       total_usage: Usage | None = None        # Running total
       tool_calls_made: int = 0
       content_preview: str | None = None      # Truncated result for display
   ```

3. **Modifying `run_tool_loop`:** Add an `on_event: Callable[[ToolLoopEvent], Awaitable[None]] | None = None` parameter. Fire events at each lifecycle point, wrapped in try/except to prevent callback errors from breaking execution:
   ```python
   async def _fire_event(on_event, event):
       if on_event is None:
           return
       try:
           await on_event(event)
       except Exception:
           logger.warning("Status callback error", exc_info=True)
   ```

   Insertion points in the existing loop:
   - Before `provider.chat()` → `LLM_CALL_START`
   - After `provider.chat()` returns → `LLM_CALL_COMPLETE` (with usage_delta)
   - Before `_handle_tool_call()` → `TOOL_CALL_START` (with tool_name)
   - After `_handle_tool_call()` returns → `TOOL_CALL_COMPLETE`
   - After loop exits → `LOOP_COMPLETE`

4. **StatusEmbed builder:**
   ```python
   @dataclass
   class StatusSnapshot:
       agent_name: str
       thread_id: int
       status: str                     # "processing", "waiting", "completed", "error", "cancelled"
       step_number: int
       current_step: str
       active_thread_count: int
       token_usage: Usage
       llm_iterations: int
       tool_calls_made: int
       tools_used: list[str]           # Unique tool names used so far
       elapsed_ms: int
       error_message: str | None = None

   def build_status_embed(snapshot: StatusSnapshot) -> discord.Embed:
       """Build a Discord embed from a status snapshot."""
       # Color: blue=processing, yellow=waiting, green=completed, red=error
       # Title: "Agent: {name} | Thread #{id}"
       # Fields: Status, Step, Threads, Tokens, Iterations, Tools, Elapsed
       ...
   ```

5. **LiveStatusView — manages one embed per thread:**
   ```python
   class LiveStatusView:
       def __init__(
           self,
           channel: discord.TextChannel,
           agent_name: str,
           thread_id: int,
           get_active_count: Callable[[], int],  # Lambda to query ThreadManager
           min_edit_interval: float = 1.5,
       ):
           self._channel = channel
           self._message: discord.Message | None = None
           self._snapshot = StatusSnapshot(...)
           self._last_edit: float = 0.0
           self._pending_update: bool = False
           self._lock = asyncio.Lock()

       async def start(self) -> None:
           """Send the initial status embed."""
           embed = build_status_embed(self._snapshot)
           self._message = await self._channel.send(embed=embed)

       async def update(self, **changes) -> None:
           """Update snapshot fields and schedule a throttled embed edit."""
           # Merge changes into self._snapshot
           # If enough time since last edit: edit now
           # Else: mark pending, schedule deferred edit

       async def finalize(self, status: str, error: str | None = None) -> None:
           """Always send the final embed regardless of throttle."""
           ...
   ```

6. **BotPresenceManager — global singleton:**
   ```python
   class BotPresenceManager:
       def __init__(self, bot: commands.Bot, debounce_seconds: float = 5.0):
           self._bot = bot
           self._active: dict[str, set[int]] = {}  # agent_name → {thread_ids}
           self._last_update: float = 0.0
           self._lock = asyncio.Lock()

       async def thread_started(self, agent_name: str, thread_id: int) -> None:
           self._active.setdefault(agent_name, set()).add(thread_id)
           await self._update_presence()

       async def thread_completed(self, agent_name: str, thread_id: int) -> None:
           self._active.get(agent_name, set()).discard(thread_id)
           if not self._active.get(agent_name):
               self._active.pop(agent_name, None)
           await self._update_presence()

       async def _update_presence(self) -> None:
           # Debounce: skip if last update was < debounce_seconds ago (schedule deferred)
           total_tasks = sum(len(t) for t in self._active.values())
           agent_count = len(self._active)
           if total_tasks > 0:
               activity = discord.Activity(
                   type=discord.ActivityType.custom,
                   name=f"Processing {total_tasks} task(s) | {agent_count} agent(s)",
               )
           else:
               activity = discord.Activity(
                   type=discord.ActivityType.custom,
                   name="Idle",
               )
           await self._bot.change_presence(activity=activity)
   ```

7. **Wiring in `bot.py`:** The runner (`_make_llm_runner`) creates a `LiveStatusView` and an `on_event` callback that bridges tool loop events to status view updates:
   ```python
   async def runner(thread: ExecutionThread) -> None:
       status_view = LiveStatusView(
           channel=message.channel,
           agent_name=agent_name,
           thread_id=thread.id,
           get_active_count=lambda: len(tm.list_active()),
       )
       await status_view.start()
       await presence_manager.thread_started(agent_name, thread.id)

       async def on_event(event: ToolLoopEvent) -> None:
           await status_view.update(
               step_number=event.iteration,
               token_usage=event.total_usage,
               tool_calls_made=event.tool_calls_made,
               # ... map event fields to snapshot fields
           )

       try:
           result = await run_tool_loop(..., on_event=on_event)
           await status_view.finalize("completed")
       except Exception as e:
           await status_view.finalize("error", error=str(e))
       finally:
           await presence_manager.thread_completed(agent_name, thread.id)
   ```

8. **Embed visual layout:**
   ```
   ┌──────────────────────────────────────────┐
   │  Agent: frontend-agent | Thread #3       │
   ├──────────────────────────────────────────┤
   │  Status:    🔄 Processing                │
   │  Step:      4 — Executing bash_execute   │
   │  Threads:   2 active                     │
   ├──────────────────────────────────────────┤
   │  Tokens:    1,247 in / 892 out           │
   │  LLM Calls: 3                            │
   │  Tools:     4 calls (bash, create_file)  │
   │  Elapsed:   12.4s                        │
   └──────────────────────────────────────────┘
   ```
   On completion:
   ```
   ┌──────────────────────────────────────────┐
   │  Agent: frontend-agent | Thread #3       │
   ├──────────────────────────────────────────┤
   │  Status:    ✅ Completed                  │
   │  Steps:     7 total                      │
   │  Threads:   1 active                     │
   ├──────────────────────────────────────────┤
   │  Tokens:    3,891 in / 2,104 out         │
   │  LLM Calls: 5                            │
   │  Tools:     6 calls (bash, create_file,  │
   │             str_replace, git_commit)      │
   │  Elapsed:   34.7s                        │
   └──────────────────────────────────────────┘
   ```

9. **Throttle implementation:** Use `asyncio.get_event_loop().time()` for monotonic timing. On `update()`, if interval elapsed → edit immediately. Otherwise set `_pending_update = True` and schedule a deferred `asyncio.create_task` that sleeps until the interval elapses, then edits if still pending. `finalize()` always cancels any pending deferred task and edits immediately.

10. **Thread metrics integration:** The `LiveStatusView.update()` should also call `thread.metrics.begin_step()` with the current step description from the event, keeping ThreadMetrics in sync with the status display. This is optional — the status view can work purely from tool loop events without touching ThreadMetrics.

11. **Error resilience:** All Discord API calls (`send`, `edit`, `change_presence`) are wrapped in try/except. A failed status update is logged at WARNING level and silently ignored. The tool loop must never be interrupted by a status display failure.

## Dependencies

- **006-execution-threads**: ThreadManager, ExecutionThread, ThreadMetrics — provides thread lifecycle and active thread counts.
- **008-llm-integration**: `run_tool_loop`, Usage, ToolLoopResult — the loop to instrument with events.
- **001-core-bot**: Bot instance for presence management.
