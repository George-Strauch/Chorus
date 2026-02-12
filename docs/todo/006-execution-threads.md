# TODO 006 — Execution Threads

> **Status:** COMPLETED

## Objective

Implement concurrent execution threads per agent. An agent can run multiple independent LLM tool loops simultaneously, each with its own conversation context. Users interact via reply-based message routing: replying to a bot message injects into that thread's context, a new (non-reply) message starts a new thread. Threads share a workspace with file-level locking to prevent conflicts.

This is foundational to how users interact with agents. Without it, the agent is single-threaded — one message at a time, blocking until done. With it, a user can say "refactor the auth module" and then immediately say "also write tests for the user model" without waiting.

## Acceptance Criteria

- Each agent maintains a `ThreadManager` that tracks active execution threads.
- A new (non-reply) message in an agent channel starts a new execution thread with a fresh context.
- A reply to a bot message routes the user's message into the thread that produced that bot message.
- Each thread has: an ID, its own message history, a status (`running`, `waiting_for_permission`, `idle`, `completed`), the Discord message IDs it has produced, a creation timestamp, and **metrics**.
- **Thread metrics** tracked per thread:
  - `elapsed_ms`: wall-clock time since thread creation
  - `step_number`: how many steps the thread has completed (each LLM call, tool execution, permission wait, etc. is a step)
  - `current_step`: human-readable description of what the thread is doing right now (e.g. "Calling LLM", "Executing bash: pytest", "Awaiting file lock on main.py", "Waiting for user permission")
  - `step_history`: ordered list of all steps the thread has gone through, each with a timestamp, duration, and description — **persisted to SQLite** so the full execution trace is available after the thread completes
- Metrics are visible via `/status` and `/thread list`.
- Multiple threads run concurrently via `asyncio.Task` — they do not block each other.
- File-level locking prevents two threads from writing the same file simultaneously. Locking is per-agent, not global. Reads do not require locks.
- `/thread list` shows all active threads for the agent in the current channel with status and summary.
- `/thread kill <thread_id>` cancels a running thread's asyncio task and cleans up.
- `/thread kill all` cancels all threads for the agent.
- Thread status (including metrics) is injected into each thread's system prompt so the LLM knows what else is happening:
  ```
  Active threads in this agent:
    Thread #1 (this thread): Refactoring auth module — step 7, 45s elapsed, currently: Executing bash: ruff check
    Thread #2: Writing user model tests — step 3, 12s elapsed, currently: Calling LLM
  ```
- Completed threads remain visible in `/thread list` for a configurable duration (default 10 minutes) then are cleaned up.
- When a thread hits an ASK permission prompt, it pauses (status = `waiting_for_permission`) and other threads continue running.
- Rate limiting: bot messages are queued with a per-channel debouncer to avoid hitting Discord's 5 messages/5 seconds limit.

## Discord API Limitations to Handle

- **Rate limits:** 5 messages per 5 seconds per channel. Multiple threads posting simultaneously will collide. Must queue outgoing messages with a per-channel rate limiter.
- **Message length:** 2000 characters. Long LLM responses or tool outputs need splitting or file attachments.
- **No persistent UI widgets:** Cannot pin a live dashboard. Use `/thread list` on demand or periodic status embeds.
- **Reply chains:** Discord's reply feature gives us routing for free — `message.reference.message_id` identifies the parent message, which maps to a thread ID.

## Tests to Write First

File: `tests/test_agent/test_threads.py` (new file)
```
# Thread lifecycle
test_new_message_creates_new_thread
test_reply_routes_to_existing_thread
test_reply_to_unknown_message_creates_new_thread
test_thread_completes_when_tool_loop_finishes
test_thread_kill_cancels_asyncio_task
test_thread_kill_all_cancels_all_threads
test_completed_thread_cleaned_up_after_timeout

# Concurrency
test_two_threads_run_concurrently
test_thread_does_not_block_other_threads
test_thread_waiting_for_permission_does_not_block_others

# File locking
test_file_lock_acquired_before_write
test_file_lock_blocks_concurrent_write_to_same_file
test_file_lock_allows_concurrent_writes_to_different_files
test_file_lock_allows_concurrent_reads
test_file_lock_released_on_tool_completion
test_file_lock_released_on_thread_kill
test_file_lock_timeout_returns_error_to_llm

# Message routing
test_message_reference_maps_to_thread_id
test_non_reply_message_starts_new_thread
test_bot_messages_tagged_with_thread_id

# Metrics
test_thread_tracks_elapsed_time
test_thread_increments_step_number_on_llm_call
test_thread_increments_step_number_on_tool_execution
test_thread_updates_current_step_description
test_step_history_records_all_steps
test_step_history_includes_timestamp_and_duration
test_step_history_persisted_to_sqlite
test_step_history_queryable_after_thread_completes
test_metrics_visible_in_thread_list
test_metrics_visible_in_status_command

# Context injection
test_thread_status_injected_into_system_prompt
test_thread_status_shows_other_active_threads
test_thread_status_includes_metrics

# Rate limiting
test_message_queue_respects_discord_rate_limit
test_messages_from_multiple_threads_interleaved_fairly
```

File: `tests/test_commands/test_thread_commands.py` (new file)
```
test_thread_list_shows_active_threads
test_thread_list_empty_when_no_threads
test_thread_kill_by_id
test_thread_kill_all
test_thread_kill_nonexistent_returns_error
```

## Implementation Notes

1. **Module location:** `src/chorus/agent/threads.py` for `ThreadManager` and `ExecutionThread`. New command file `src/chorus/commands/thread_commands.py`.

2. **ExecutionThread dataclass:**
   ```python
   @dataclass
   class ThreadStep:
       step_number: int
       description: str            # "Calling LLM", "Executing bash: pytest", "Awaiting file lock on main.py"
       started_at: datetime
       ended_at: datetime | None = None
       duration_ms: int | None = None

   @dataclass
   class ThreadMetrics:
       created_at: datetime
       step_number: int = 0
       current_step: str = "Starting"
       step_history: list[ThreadStep] = field(default_factory=list)

       @property
       def elapsed_ms(self) -> int:
           return int((datetime.now() - self.created_at).total_seconds() * 1000)

       def begin_step(self, description: str) -> None:
           """End the current step (if any) and start a new one."""
           if self.step_history and self.step_history[-1].ended_at is None:
               step = self.step_history[-1]
               step.ended_at = datetime.now()
               step.duration_ms = int((step.ended_at - step.started_at).total_seconds() * 1000)
           self.step_number += 1
           self.current_step = description
           self.step_history.append(ThreadStep(
               step_number=self.step_number,
               description=description,
               started_at=datetime.now(),
           ))

   @dataclass
   class ExecutionThread:
       id: int                              # Auto-incrementing per agent
       agent_name: str
       messages: list[dict]                 # This thread's conversation history
       status: ThreadStatus                 # running, waiting_for_permission, idle, completed
       task: asyncio.Task | None            # The running tool loop task
       discord_message_ids: list[int]       # Messages this thread produced (for reply routing)
       metrics: ThreadMetrics               # Timing, steps, current activity
       summary: str | None = None           # Brief description, set by LLM after first response
   ```

3. **ThreadManager:**
   ```python
   class ThreadManager:
       def __init__(self, agent_name: str) -> None:
           self._threads: dict[int, ExecutionThread] = {}
           self._message_to_thread: dict[int, int] = {}  # discord_msg_id -> thread_id
           self._file_locks: dict[Path, asyncio.Lock] = {}
           self._next_id: int = 1

       def create_thread(self, initial_message: dict) -> ExecutionThread: ...
       def route_message(self, discord_message: Message) -> ExecutionThread | None: ...
       def get_thread(self, thread_id: int) -> ExecutionThread | None: ...
       async def kill_thread(self, thread_id: int) -> bool: ...
       async def kill_all(self) -> int: ...
       def list_active(self) -> list[ExecutionThread]: ...
       def register_bot_message(self, discord_msg_id: int, thread_id: int) -> None: ...
       async def acquire_file_lock(self, path: Path, timeout: float = 30) -> bool: ...
       def release_file_lock(self, path: Path) -> None: ...
   ```

4. **Message routing logic in `bot.py on_message`:**
   ```python
   async def on_message(self, message: Message) -> None:
       if message.author == self.user:
           return
       agent = self.agent_manager.get_agent_by_channel(message.channel.id)
       if not agent:
           await self.process_commands(message)
           return

       thread_manager = agent.thread_manager

       # Reply → route to existing thread
       if message.reference and message.reference.message_id:
           thread = thread_manager.route_message(message)

       # Non-reply → new thread
       if thread is None:
           thread = thread_manager.create_thread(user_message_dict(message))

       # Inject message into thread's context and (re)start the tool loop
       thread.messages.append(user_message_dict(message))
       if thread.status != ThreadStatus.RUNNING:
           thread.task = asyncio.create_task(
               run_thread(agent, thread, message.channel)
           )
   ```

5. **File locking — per-agent, per-file `asyncio.Lock`:**
   ```python
   async def acquire_file_lock(self, path: Path, timeout: float = 30) -> bool:
       lock = self._file_locks.setdefault(path.resolve(), asyncio.Lock())
       try:
           await asyncio.wait_for(lock.acquire(), timeout=timeout)
           return True
       except asyncio.TimeoutError:
           return False  # Return error to LLM: "file locked by another thread"
   ```
   File tools (`create_file`, `str_replace`) must acquire the lock before writing and release after. `view` does not lock (reads are safe). If the lock times out, the tool returns an error message to the LLM so it can retry or work on something else.

6. **SQLite schema for thread step history:**
   ```sql
   CREATE TABLE IF NOT EXISTS thread_steps (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       agent_name TEXT NOT NULL,
       thread_id INTEGER NOT NULL,
       step_number INTEGER NOT NULL,
       description TEXT NOT NULL,
       started_at TEXT NOT NULL,
       ended_at TEXT,
       duration_ms INTEGER
   );

   CREATE INDEX IF NOT EXISTS idx_thread_steps_agent_thread
       ON thread_steps(agent_name, thread_id);
   ```
   Steps are written to SQLite as they complete (not just at thread end) so they survive crashes and are queryable immediately. On thread completion, any in-progress step is finalized.

7. **`/status` and `/thread list` display:**
   `/status` shows an embed per active thread:
   ```
   Agent: frontend-agent
   ─────────────────────
   Thread #1: Refactoring auth module
     Status:  running
     Elapsed: 45s
     Step:    7 — Executing bash: ruff check

   Thread #2: Writing user model tests
     Status:  waiting_for_permission
     Elapsed: 12s
     Step:    3 — Waiting for user permission: tool:bash:rm -rf node_modules
   ─────────────────────
   ```

   `/thread list` includes the same metrics in a compact table. `/thread history <thread_id>` (or `/thread steps <thread_id>`) shows the full step history for a completed thread:
   ```
   Thread #1 — Refactoring auth module (completed, 47s total)
     Step 1:  Calling LLM (2100ms)
     Step 2:  Executing bash: ls src/ (340ms)
     Step 3:  Calling LLM (1800ms)
     Step 4:  Executing file:str_replace auth.py (12ms)
     Step 5:  Calling LLM (2400ms)
     Step 6:  Executing bash: pytest tests/test_auth.py (8200ms)
     Step 7:  Calling LLM (1900ms)
   ```

8. **Thread status in system prompt:** Before each LLM call, inject a summary of other active threads including metrics:
   ```python
   def build_thread_status(thread_manager: ThreadManager, current_thread_id: int) -> str:
       lines = ["Active threads:"]
       for t in thread_manager.list_active():
           marker = " (this thread)" if t.id == current_thread_id else ""
           elapsed = f"{t.metrics.elapsed_ms / 1000:.0f}s"
           lines.append(
               f"  #{t.id}{marker}: {t.summary or 'Starting...'} "
               f"— step {t.metrics.step_number}, {elapsed} elapsed, "
               f"currently: {t.metrics.current_step} [{t.status.value}]"
           )
       return "\n".join(lines)
   ```

9. **Metrics integration in tool loop:** The tool loop calls `metrics.begin_step()` at each phase:
   ```python
   # In the tool loop, per iteration:
   thread.metrics.begin_step(f"Calling LLM")
   response = await provider.chat(...)

   for tc in response.tool_calls:
       thread.metrics.begin_step(f"Executing {tc.name}: {tc.arguments_summary}")
       result = await tool.execute(...)
       # Step is persisted to SQLite here
       await persist_step(db, thread)

   # On permission ASK:
   thread.metrics.begin_step(f"Waiting for user permission: {action_string}")
   ```

10. **Rate-limited message sending:**
   ```python
   class ChannelMessageQueue:
       """Per-channel queue that respects Discord's 5msg/5s rate limit."""
       def __init__(self, channel: TextChannel) -> None: ...
       async def send(self, content: str, **kwargs) -> Message: ...
   ```
   All bot messages go through this queue. The queue sends up to 5 messages, then waits for the rate limit window to reset. Messages from different threads are interleaved fairly (round-robin or FIFO).

11. **Thread cleanup:** A background task periodically sweeps completed threads older than the configurable retention period. Thread message history can optionally be appended to the agent's session for context continuity.

12. **Interjection into a running thread:** When a user replies to a running thread's message, the new user message is appended to the thread's `messages` list. If the tool loop is mid-iteration (waiting on LLM or tool execution), the injected message will be seen on the next iteration. If the thread is between iterations, it triggers a new iteration immediately. The LLM sees the interjection as a new user message in the conversation and adapts.

13. **Edge case — reply to a completed thread:** If the user replies to a message from a completed thread, start a new thread but seed it with the completed thread's context (or a summary of it). This provides continuity without resurrecting a dead thread.

## Dependencies

- **005-bash-execution**: Threads dispatch tool calls including bash.
- **007-context-management**: Per-thread context, session save must handle multi-thread state.
- **008-llm-integration**: Each thread runs a tool loop instance.

## Interaction with Other TODOs

- **007 (Context):** Session save/restore must account for multiple threads. On idle timeout, all threads are summarized and saved. On restore, the single-thread model is restored (threads are ephemeral, not persisted across sessions).
- **008 (Tool Loop):** The tool loop function is called per-thread. It must accept a `thread_id` parameter for file locking and status tracking.
- **004 (File Tools):** `create_file` and `str_replace` must acquire file locks via the thread manager before writing.
