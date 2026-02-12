# TODO 007 — Context Management

## Objective

Implement session save/restore, idle timer, natural language summaries, and context clearing for agents. Agents accumulate conversation history during a work session. When a session ends (manually or via idle timeout), the conversation is summarized in natural language, saved to the agent's `sessions/` directory and indexed in SQLite, and the context is cleared. Users can restore previous sessions to continue work.

## Acceptance Criteria

- `/context save [description]` saves the current session: serializes conversation messages to JSON in `~/.chorus-agents/agents/<name>/sessions/<timestamp>_<slug>.json`, generates a natural language summary via the LLM, stores metadata (timestamp, summary, description, message count, token estimate) in SQLite, and clears the in-memory context.
- `/context clear` clears the in-memory conversation without saving.
- `/context history [agent_name]` lists saved sessions with timestamp, description, summary, and message count. Paginated if more than 10 sessions.
- `/context restore <session_id>` loads a saved session's messages back into the in-memory context.
- **Idle timer:** Each agent has a configurable idle timeout (default 30 minutes). After the timeout elapses with no messages in the channel, the bot automatically triggers a save (with an auto-generated description), posts a summary message to the channel, and clears context.
- The idle timer resets on every message in the agent's channel.
- Session files use the naming format: `YYYY-MM-DD_<slug>.json` where slug is derived from the description or auto-generated.
- In-memory context is stored as a list of message dicts (role, content, tool_use, tool_result) in the `Agent` runtime state — not in SQLite.
- Context module exposes: `SessionManager` class with `save()`, `restore()`, `clear()`, `list_sessions()`, `get_summary()`.
- Summary generation uses the agent's configured LLM provider to produce a 2-4 sentence summary of the session.
- Old sessions are retained indefinitely (no auto-cleanup — agents may need to reference past work).

## Tests to Write First

File: `tests/test_agent/test_context.py`
```
# Save
test_save_session_creates_json_file
test_save_session_stores_metadata_in_sqlite
test_save_session_generates_summary
test_save_session_clears_context_after_save
test_save_session_filename_format
test_save_session_with_custom_description
test_save_session_empty_context_raises

# Restore
test_restore_session_loads_messages
test_restore_session_sets_context
test_restore_session_nonexistent_id_raises
test_restore_session_corrupted_file_raises

# Clear
test_clear_context_empties_messages
test_clear_context_does_not_save

# History
test_list_sessions_returns_all_sessions
test_list_sessions_ordered_by_timestamp_desc
test_list_sessions_empty_returns_empty
test_list_sessions_includes_summary

# Idle timer
test_idle_timer_triggers_after_timeout
test_idle_timer_resets_on_message
test_idle_timer_auto_save_creates_session
test_idle_timer_posts_summary_to_channel
test_idle_timer_configurable_timeout
test_idle_timer_no_trigger_when_context_empty

# Summary generation
test_generate_summary_calls_llm
test_generate_summary_returns_short_text
test_generate_summary_handles_llm_failure_gracefully

# Serialization
test_session_json_roundtrip
test_session_includes_tool_use_messages
test_session_slug_generation_from_description
test_session_slug_generation_auto
```

File: `tests/test_commands/test_context_commands.py`
```
test_context_save_command_calls_session_manager
test_context_clear_command_calls_clear
test_context_history_command_returns_embed
test_context_restore_command_loads_session
test_context_save_command_no_active_agent_in_channel
```

## Implementation Notes

1. **Module location:** `src/chorus/agent/context.py` for `SessionManager`, `src/chorus/commands/context_commands.py` for slash commands.

2. **In-memory context storage:** The runtime `Agent` object (not the on-disk `agent.json`) holds a `messages: list[dict]` field. This list is the conversation history fed to the LLM on each call. It is never persisted to `agent.json` — it lives only in memory, and is serialized to `sessions/` on save.

3. **SessionManager class:**
   ```python
   class SessionManager:
       def __init__(self, agent_dir: Path, db: Database) -> None:
           self._sessions_dir = agent_dir / "sessions"
           self._db = db

       async def save(self, messages: list[dict], description: str | None = None,
                      summarizer: Callable | None = None) -> SessionMetadata: ...
       async def restore(self, session_id: str) -> list[dict]: ...
       def clear(self) -> list[dict]:  # Returns empty list for assignment
           return []
       async def list_sessions(self, limit: int = 50) -> list[SessionMetadata]: ...
   ```

4. **Session file format:**
   ```json
   {
     "session_id": "2026-02-12_auth-refactor",
     "timestamp": "2026-02-12T14:30:00Z",
     "description": "Working on auth refactor",
     "summary": "Implemented JWT token validation and refresh flow...",
     "message_count": 47,
     "token_estimate": 12500,
     "messages": [...]
   }
   ```

5. **Slug generation:** From description: lowercase, replace spaces with hyphens, strip non-alphanumeric, truncate to 40 chars. If no description, use a 6-char hash of the first message content. If collision, append `-2`, `-3`, etc.

6. **Idle timer implementation:** Use `asyncio.Task` per agent. The task sleeps for the timeout duration. On any message in the agent's channel, cancel the current task and start a new one.
   ```python
   class IdleTimer:
       def __init__(self, timeout: float, callback: Callable[[], Awaitable[None]]) -> None:
           self._timeout = timeout
           self._task: asyncio.Task | None = None
           self._callback = callback

       def reset(self) -> None:
           if self._task:
               self._task.cancel()
           self._task = asyncio.create_task(self._run())

       async def _run(self) -> None:
           await asyncio.sleep(self._timeout)
           await self._callback()

       def stop(self) -> None:
           if self._task:
               self._task.cancel()
               self._task = None
   ```

7. **Summary generation:** Call the agent's configured LLM with a system prompt like: "Summarize this conversation in 2-4 sentences. Focus on what was accomplished, decisions made, and any unfinished work." Pass the full conversation as the user message. If the LLM call fails (network error, etc.), save the session with `summary: "(summary generation failed)"` rather than failing the entire save.

8. **SQLite schema for sessions:**
   ```sql
   CREATE TABLE sessions (
       session_id TEXT PRIMARY KEY,
       agent_name TEXT NOT NULL,
       timestamp TEXT NOT NULL,
       description TEXT,
       summary TEXT,
       message_count INTEGER,
       token_estimate INTEGER,
       file_path TEXT NOT NULL
   );
   ```

9. **Token estimation:** Use a rough heuristic: `sum(len(msg["content"]) for msg in messages) // 4`. This doesn't need to be exact — it's for informational display only.

## Dependencies

- **002-agent-manager**: Agent directory structure (sessions/ directory).
- **008-llm-integration**: Summary generation requires the LLM client. However, the `SessionManager` accepts a `summarizer` callable, so it can be tested with a mock. The actual wiring to the LLM happens when 008 is complete.
