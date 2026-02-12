# TODO 007 — Context Management

> **Status:** PENDING

## Objective

Implement the context system for agents. Context is built from a **rolling window** of persisted chat messages — by default the last 24 hours, or everything since the most recent `/context clear`, whichever is more recent. Messages are persisted to SQLite so context survives bot restarts. Each LLM call also receives a preamble with active thread status and running process summaries.

## Core Context Model

**Rolling window:** Every message (user, assistant, tool_use, tool_result) in an agent's channel is persisted to SQLite with a timestamp. When building context for an LLM call, we query:

```sql
SELECT * FROM messages
WHERE agent_name = ? AND timestamp > ?
ORDER BY timestamp ASC
```

Where `?` is `max(last_clear_time, now - rolling_window)`.

- `rolling_window`: configurable per-agent, default 86400 seconds (1 day).
- `last_clear_time`: set by `/context clear`, stored per-agent in the `agents` table or a separate setting. Default is agent creation time.
- `/context clear` does NOT delete messages from the database — it just advances the `last_clear_time` marker. Old messages remain queryable for history but are excluded from the active context window.

**Context assembly for an LLM call:**
```
1. System prompt (from agent.json)
2. Agent docs/ contents
3. Thread & process status summary (from ThreadManager / ProcessManager)
4. Rolling window messages (for this thread's conversation)
```

## Acceptance Criteria

- All messages in agent channels are persisted to SQLite with: agent_name, thread_id, role, content, tool_calls, timestamp.
- Context for an LLM call = messages since `max(last_clear_time, now - rolling_window)` for the relevant thread.
- `/context clear` advances the agent's `last_clear_time` to now. New threads see only messages after this point.
- `/context save [description]` snapshots the current rolling window to a session JSON file with an LLM-generated summary. Does NOT clear context (that's what `/context clear` is for).
- `/context history [agent_name]` lists saved session snapshots with timestamp, description, summary, and message count.
- `/context restore <session_id>` loads a saved session's messages back into the rolling window (inserts them with current timestamps or a special marker).
- `rolling_window` duration is configurable via `/settings` (default 1 day).
- On bot restart, context is rebuilt from persisted messages — no data loss.
- Old messages beyond the rolling window are retained in the database indefinitely for history queries.

## Tests to Write First

File: `tests/test_agent/test_context.py`
```
# Message persistence
test_message_persisted_to_sqlite
test_message_includes_agent_name_and_thread_id
test_message_includes_timestamp
test_messages_queryable_by_agent

# Rolling window
test_rolling_window_returns_messages_within_window
test_rolling_window_excludes_old_messages
test_rolling_window_default_is_one_day
test_rolling_window_configurable

# Clear
test_clear_advances_last_clear_time
test_clear_excludes_messages_before_clear_time
test_clear_does_not_delete_messages_from_db
test_messages_after_clear_are_included
test_multiple_clears_use_most_recent

# Context assembly
test_context_includes_system_prompt
test_context_includes_agent_docs
test_context_includes_thread_status
test_context_includes_rolling_window_messages
test_context_window_bounded_by_clear_time_or_rolling_window

# Save (snapshot)
test_save_creates_session_json_file
test_save_stores_metadata_in_sqlite
test_save_generates_summary
test_save_does_not_clear_context
test_save_filename_format
test_save_with_custom_description

# Restore
test_restore_loads_saved_messages
test_restore_nonexistent_id_raises

# History
test_list_sessions_returns_all_snapshots
test_list_sessions_ordered_by_timestamp_desc

# Bot restart
test_context_survives_restart_via_sqlite
test_clear_time_survives_restart

# Summary generation
test_generate_summary_calls_llm
test_generate_summary_returns_short_text
test_generate_summary_handles_llm_failure_gracefully
```

File: `tests/test_commands/test_context_commands.py`
```
test_context_clear_command_advances_clear_time
test_context_save_command_creates_snapshot
test_context_history_command_returns_embed
test_context_restore_command_loads_session
test_context_commands_no_active_agent_in_channel
```

## Implementation Notes

1. **Module location:** `src/chorus/agent/context.py` for `ContextManager`, `src/chorus/commands/context_commands.py` for slash commands.

2. **New SQLite table for messages:**
   ```sql
   CREATE TABLE IF NOT EXISTS messages (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       agent_name TEXT NOT NULL,
       thread_id INTEGER,
       role TEXT NOT NULL,           -- "user", "assistant", "tool_use", "tool_result"
       content TEXT,
       tool_calls TEXT,              -- JSON-serialized tool call data, if any
       tool_call_id TEXT,            -- For tool_result messages
       timestamp TEXT NOT NULL,
       discord_message_id INTEGER    -- For reply routing back to threads
   );

   CREATE INDEX IF NOT EXISTS idx_messages_agent_time
       ON messages(agent_name, timestamp);
   ```

3. **Agent clear time:** Add a `last_clear_time` column to the `agents` table (or store in `settings`). Default to agent `created_at`.

4. **ContextManager class:**
   ```python
   class ContextManager:
       def __init__(self, agent_name: str, db: Database, rolling_window: int = 86400) -> None:
           self._agent_name = agent_name
           self._db = db
           self._rolling_window = rolling_window

       async def persist_message(self, thread_id: int, role: str, content: str | None,
                                  tool_calls: list | None = None, **kwargs) -> None:
           """Store a message in SQLite."""
           ...

       async def get_context(self, thread_id: int) -> list[dict]:
           """Return messages within the rolling window for this thread."""
           cutoff = max(self._last_clear_time, now - self._rolling_window)
           ...

       async def clear(self) -> None:
           """Advance last_clear_time to now."""
           ...

       async def save_snapshot(self, description: str | None = None,
                                summarizer: Callable | None = None) -> SessionMetadata:
           """Save current window to a session file."""
           ...

       async def list_snapshots(self, limit: int = 50) -> list[SessionMetadata]: ...
       async def restore_snapshot(self, session_id: str) -> None: ...
   ```

5. **Context assembly helper:**
   ```python
   async def build_llm_context(
       agent: Agent,
       thread_id: int,
       context_manager: ContextManager,
       thread_manager: ThreadManager,
   ) -> list[dict]:
       """Assemble the full context for an LLM call."""
       messages = []
       # 1. System prompt (handled separately by provider)
       # 2. Agent docs
       messages.append({"role": "user", "content": read_agent_docs(agent)})
       # 3. Thread/process status
       status = build_thread_status(thread_manager, thread_id)
       if status:
           messages.append({"role": "user", "content": status})
       # 4. Rolling window messages for this thread
       messages.extend(await context_manager.get_context(thread_id))
       return messages
   ```

6. **Session snapshot file format:**
   ```json
   {
     "session_id": "uuid",
     "timestamp": "2026-02-12T14:30:00Z",
     "description": "Working on auth refactor",
     "summary": "Implemented JWT token validation and refresh flow...",
     "message_count": 47,
     "window_start": "2026-02-12T00:00:00Z",
     "window_end": "2026-02-12T14:30:00Z",
     "messages": [...]
   }
   ```

7. **Summary generation:** Call the agent's configured LLM with a system prompt like: "Summarize this conversation in 2-4 sentences. Focus on what was accomplished, decisions made, and any unfinished work." If the LLM call fails, save with `summary: "(summary generation failed)"`.

8. **Token estimation:** Rough heuristic: `sum(len(msg.get("content", "")) for msg in messages) // 4`. Informational only.

## Thread-Awareness (TODO 006 interaction)

Context management accounts for concurrent execution threads:

- Each message is tagged with a `thread_id` in SQLite.
- `get_context(thread_id)` returns only messages for that thread within the rolling window.
- Thread status (all active threads) is injected as a preamble into every LLM call so each thread knows what else is happening.
- `/context clear` affects ALL threads — it advances the agent-level clear time.
- `/context save` snapshots ALL threads' messages within the current window.

## Dependencies

- **002-agent-manager**: Agent directory structure (sessions/ directory).
- **006-execution-threads**: Per-thread context, thread status injection.
- **008-llm-integration**: Summary generation requires the LLM client. However, the `ContextManager` accepts a `summarizer` callable, so it can be tested with a mock.
