# Architecture

This document describes the system design of Chorus, a Discord bot that turns channels into autonomous AI agents. Each agent has its own workspace, tools, permission profile, and LLM configuration. Agents are general purpose -- what they do is defined entirely by their system prompt and tool access, not by any hardcoded domain logic.

---

## High-Level Architecture

```
Discord Server
  |
  |  (slash commands, messages)
  v
+------------------+
|    bot.py        |  Discord gateway, slash command dispatch
+------------------+
  |
  v
+------------------+       +---------------------+
|  AgentManager    | <---> |  SQLite (db.py)     |
|  (manager.py)    |       |  sessions, audit,   |
+------------------+       |  agent state,       |
  |                        |  settings           |
  |  creates/manages       +---------------------+
  v
+------------------+
|  Agent Instance  |  Per-channel: config, context, workspace, git repo
+------------------+
  |
  |  dispatches tool calls
  v
+------------------+       +---------------------+
|  Tool Loop       | <---> |  Tool Registry      |
|  (tool_loop.py)  |       |  (registry.py)      |
+------------------+       +---------------------+
  |                            |
  |  LLM requests              |  filters tools by PermissionProfile
  v                            v
+------------------+       +---------------------+
|  LLM Providers   |       |  Permission Engine  |
|  (providers.py)  |       |  (engine.py)        |
+------------------+       +---------------------+
  |
  |  Anthropic SDK / OpenAI SDK
  v
+------------------+
|  Claude / GPT    |
+------------------+
```

### Component Summary

| Component | File | Responsibility |
|---|---|---|
| **Bot** | `bot.py` | Discord gateway connection, slash command registration, message routing to agents |
| **Agent Manager** | `agent/manager.py` | Agent lifecycle: create from template, destroy, list, configure |
| **Agent Directory** | `agent/directory.py` | Filesystem operations for agent workspaces under `~/.chorus-agents/` |
| **Context Manager** | `agent/context.py` | Session save/restore, idle timer, NL summaries, context window tracking |
| **Self-Edit** | `agent/self_edit.py` | Agent self-modification of config, system prompt, docs, permissions |
| **Permission Engine** | `permissions/engine.py` | Regex-based allow/ask/deny matching for action strings |
| **Tool Registry** | `tools/registry.py` | Registers all tools, filters available tools per agent based on permissions |
| **File Ops** | `tools/file_ops.py` | `create_file`, `str_replace`, `view` -- all jailed to agent workspace |
| **Bash** | `tools/bash.py` | Sandboxed command execution with cwd set to agent workspace |
| **Execution Threads** | `agent/threads.py` | Concurrent execution threads per agent, reply-based routing, file locking |
| **LLM Providers** | `llm/providers.py` | Unified `LLMProvider` protocol, Anthropic and OpenAI implementations |
| **Model Discovery** | `llm/discovery.py` | Validate API keys on startup, discover and cache available models |
| **Tool Loop** | `llm/tool_loop.py` | Agentic loop: message -> LLM -> tool_use -> execute -> result -> LLM -> ... |
| **Config** | `config.py` | Bot configuration, environment variables, server-wide defaults |
| **Models** | `models.py` | Data models: Agent, PermissionProfile, Session, etc. |
| **Storage** | `storage/db.py` | Async SQLite via aiosqlite: sessions, audit log, agent registry, settings |
| **Commands** | `commands/*.py` | Slash command implementations for `/agent`, `/settings`, `/context`, `/tasks`, `/status` |

---

## Data Flow

### User Message to Agent Response

The primary data flow is a user sending a message in a Discord channel that is bound to an agent. Messages are routed to execution threads based on whether they are replies.

```
1. User types message in #frontend-agent channel
                    |
2. Discord gateway delivers message to bot.py
                    |
3. bot.py looks up which agent owns channel_id
   (AgentManager.get_agent_by_channel)
                    |
4. Message routing (ThreadManager):
   - If message is a reply to a bot message → route to the thread
     that produced that bot message
   - If message is not a reply → create a new execution thread
                    |
5. Thread's context is prepared:
   - System prompt + docs/ contents
   - Summary of other active threads and running processes
   - This thread's conversation history
   - The new user message
                    |
6. Thread's tool loop runs (as an asyncio.Task):
   tool_loop.py calls LLM provider with context + available tools
                    |
7. LLM returns one of:
   a) text response         -> send to Discord channel, done
   b) tool_use request(s)   -> continue to step 8
                    |
8. For each tool_use in response:
   a) Build action string: tool:<tool_name>:<detail>
   b) Check PermissionEngine:
      - ALLOW  -> execute immediately
      - ASK    -> post confirmation prompt in Discord, wait for user
                  (thread status = waiting_for_permission, other threads continue)
      - DENY   -> return denial message to LLM
   c) For file writes: acquire file lock (other threads wait or get error)
   d) Execute tool, capture result, release lock
   e) Append tool_result to context
                    |
9. Send updated context (now including tool results) back to LLM
   -> Go to step 7 (loop until LLM returns text response with no tool_use)
                    |
10. Final text response is sent to Discord channel (via rate-limited queue)
    Bot message is tagged with thread ID for future reply routing
                    |
11. Thread status set to completed
    Context manager updates idle timer (resets for the whole agent)
    - On timeout: summarize all threads, save session, clear context, notify channel
```

### Slash Command Flow

Slash commands follow a different path -- they do not enter the tool loop. They are handled directly by the command modules:

```
1. User invokes /agent init my-agent
                    |
2. Discord delivers interaction to bot.py
                    |
3. bot.py routes to agent_commands.py handler
                    |
4. Handler calls AgentManager.create_agent("my-agent", ...)
   - Copies template/ directory
   - Initializes git repo in workspace/
   - Merges config overrides into agent.json
   - Registers in SQLite
   - Creates Discord channel
                    |
5. Returns ephemeral confirmation message to user
```

---

## Key Components in Detail

### Bot (`bot.py`)

The entrypoint. Connects to Discord, registers slash commands via `discord.py`'s `app_commands` system, and routes incoming messages to the correct agent. It holds a reference to `AgentManager` and the `LLMProviders` registry.

Key responsibilities:
- Establish Discord gateway connection
- Register all slash command groups (`/agent`, `/settings`, `/context`, `/tasks`, `/status`)
- Listen for messages in agent-bound channels and forward them to the tool loop
- Handle graceful shutdown (save all active sessions)

### Agent Manager (`agent/manager.py`)

Central orchestrator for agent lifecycle. Maintains an in-memory registry of active agents (keyed by channel_id and agent name) backed by SQLite for persistence.

Key operations:
- `create_agent(name, system_prompt, model, permissions)` -- full creation flow from template
- `destroy_agent(name, keep_files)` -- deregister, optionally delete workspace
- `list_agents()` -- enumerate all agents with status
- `configure_agent(name, key, value)` -- update agent.json fields
- `get_agent_by_channel(channel_id)` -- lookup for message routing

### Agent Directory (`agent/directory.py`)

Manages the filesystem layout under `~/.chorus-agents/`. Handles:
- Copying the template directory to create new agent workspaces
- Path resolution and jail checks (prevent path traversal outside agent workspace)
- Reading and writing `agent.json`
- Enumerating agent directories

### Context Manager (`agent/context.py`)

Manages conversation context via a **rolling window** of persisted messages:
- All messages (user, assistant, tool_use, tool_result) are persisted to SQLite with timestamps. Context survives bot restarts.
- **Rolling window:** context = messages since `max(last_clear_time, now - rolling_window)`. Default window is 24 hours.
- `/context clear` advances the `last_clear_time` marker. Messages before this point are excluded from active context but remain in the database for history.
- `/context save [description]` snapshots the current window to a session JSON file with an LLM-generated summary. Does not clear.
- `/context restore <session_id>` loads a saved snapshot back into the context.
- Context assembly for each LLM call: system prompt + agent docs + thread/process status + rolling window messages for the current thread.

### Self-Edit (`agent/self_edit.py`)

Implements the tools that allow an agent to modify its own configuration. These are registered as tools in the tool registry and go through the permission engine like any other tool.

Self-edit tools:
- `self_edit_system_prompt(new_prompt)` -- updates `agent.json` system_prompt field
- `self_edit_docs(path, content)` -- writes to files in the agent's `docs/` directory
- `self_edit_permissions(new_profile)` -- changes the agent's permission profile (gated by invoking user's Discord role)
- `self_edit_model(new_model)` -- switches the agent's LLM model

All self-edits are logged to the SQLite audit table.

### Permission Engine (`permissions/engine.py`)

See [PERMISSIONS.md](PERMISSIONS.md) for the full specification. In brief: every tool invocation is converted to an action string (`tool:<name>:<detail>`), then matched against the agent's permission profile in order: allow patterns -> ask patterns -> implicit DENY.

### Tool Registry (`tools/registry.py`)

Central registry of all available tools. Each tool is defined with:
- A name (used in action strings and LLM tool schemas)
- A JSON Schema for parameters (sent to LLM as tool definition)
- An async execution function
- A function to build the action string detail from the parameters

When building the tool list for an LLM call, the registry filters tools based on the agent's permission profile -- tools that would be universally denied are omitted from the schema sent to the LLM.

### Tool Implementations

**File Ops (`tools/file_ops.py`):**
- `create_file(path, content)` -- create or overwrite a file in the agent workspace
- `str_replace(path, old_str, new_str)` -- find-and-replace within a file (exact string match)
- `view(path, offset, limit)` -- read file contents with optional line range

All paths are resolved relative to the agent's `workspace/` directory. Path traversal prevention: resolve the absolute path, verify it starts with the workspace prefix. Reject if it escapes.

**Bash (`tools/bash.py`):**
- `bash(command, timeout)` -- execute a shell command with cwd set to agent workspace
- Returns stdout, stderr, and exit code
- Configurable timeout (default from bot config)
- No chroot or namespace isolation -- Docker is the outer security boundary

### Execution Threads (`agent/threads.py`)

Agents support **concurrent execution threads**. Each thread is an independent LLM tool loop with its own conversation context, running as an `asyncio.Task`. Multiple threads share the agent's workspace.

**Message routing — reply-based:**
- A new (non-reply) message in the agent channel starts a new execution thread.
- A reply to a bot message routes the user's message into the thread that produced it.
- This uses Discord's native reply feature (`message.reference.message_id`).

**File locking:**
- Threads share a workspace. Concurrent writes to the same file would corrupt it.
- File tools acquire a per-file `asyncio.Lock` before writing. Reads do not lock.
- If a lock cannot be acquired within a timeout (default 30s), the tool returns an error to the LLM so it can retry or work on something else.

**Context awareness:**
- Each thread's system prompt includes a summary of other active threads so the LLM can avoid conflicts and coordinate.

**Thread lifecycle:** created on message → running tool loop → completed on final response or killed via `/thread kill`.

### LLM Providers (`llm/providers.py`)

Defines the `LLMProvider` protocol and concrete implementations for Anthropic and OpenAI:

```python
class LLMProvider(Protocol):
    name: str

    async def chat(
        self,
        model: str,
        messages: list[Message],
        tools: list[ToolSchema],
        system: str | None = None,
    ) -> LLMResponse: ...

    async def validate_key(self) -> bool: ...

    async def list_models(self) -> list[str]: ...
```

The `LLMResponse` contains:
- `content`: the text response (if any)
- `tool_calls`: list of tool use requests (if any)
- `stop_reason`: why the model stopped (`end_turn`, `tool_use`, `max_tokens`)
- `usage`: token counts for tracking

**AnthropicProvider** wraps the `anthropic` SDK. System prompt is sent via the `system` parameter. Tools use Anthropic's native tool_use format.

**OpenAIProvider** wraps the `openai` SDK. System prompt is sent as the first message with role `system`. Tools use OpenAI's function calling format. Response tool calls are normalized to the common `LLMResponse` format.

### Model Discovery (`llm/discovery.py`)

On bot startup (and on `/settings validate-keys`):
1. For each configured API key, call the provider's `validate_key()` method
2. For valid keys, call `list_models()` to get available models
3. Cache the result in `~/.chorus-agents/available_models.json`
4. This cached list is used by `/settings model` for validation and autocomplete

### Tool Loop (`llm/tool_loop.py`)

The agentic execution loop. This is the heart of the system:

```python
async def run_tool_loop(
    agent: Agent,
    messages: list[Message],
    providers: dict[str, LLMProvider],
    tools: list[Tool],
    permission_engine: PermissionEngine,
    discord_channel: TextChannel,
) -> str:
    """
    Run the agentic tool loop until the LLM produces a final text response.
    """
    while True:
        # 1. Call LLM with current context + tool schemas
        provider = providers[agent.provider_name]
        response = await provider.chat(
            model=agent.model,
            messages=messages,
            tools=[t.schema for t in tools],
            system=agent.system_prompt,
        )

        # 2. If no tool calls, return the text response
        if not response.tool_calls:
            return response.content

        # 3. Process each tool call
        for tool_call in response.tool_calls:
            tool = tool_registry.get(tool_call.name)
            action_string = tool.build_action_string(tool_call.params)

            # 4. Permission check
            decision = permission_engine.check(agent.permissions, action_string)

            if decision == Decision.ALLOW:
                result = await tool.execute(agent, tool_call.params)
            elif decision == Decision.ASK:
                approved = await ask_user_confirmation(discord_channel, tool_call)
                if approved:
                    result = await tool.execute(agent, tool_call.params)
                else:
                    result = ToolResult(error="User denied this action.")
            else:  # DENY
                result = ToolResult(error=f"Permission denied: {action_string}")

            # 5. Append assistant message with tool_use + tool_result to context
            messages.append(assistant_tool_use_message(tool_call))
            messages.append(tool_result_message(tool_call.id, result))

        # 6. Loop back to step 1 with updated context
```

Key behaviors:
- The loop continues until the LLM returns a response with no tool calls (stop_reason = `end_turn`)
- Multiple tool calls in a single LLM response are executed sequentially (to maintain ordering and allow early abort on denial)
- ASK decisions post a confirmation prompt in the Discord channel with the action details and wait for user reaction or reply
- All tool executions (allow, ask-approved, deny) are logged to the SQLite audit table
- If the LLM hits `max_tokens`, the loop appends the partial response and continues

---

## Storage Layer

### SQLite Database (`storage/db.py`)

Located at `~/.chorus-agents/db/chorus.db`. Uses `aiosqlite` for async access. All database operations are in a single module to centralize schema and migrations.

**Tables:**

```sql
-- Agent registry
CREATE TABLE agents (
    name TEXT PRIMARY KEY,
    channel_id INTEGER UNIQUE NOT NULL,
    guild_id INTEGER NOT NULL,
    model TEXT,
    permissions TEXT NOT NULL DEFAULT 'standard',
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'  -- active, destroyed
);

-- Saved sessions (context snapshots)
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,           -- UUID
    agent_name TEXT NOT NULL REFERENCES agents(name),
    description TEXT,              -- NL summary or user-provided label
    saved_at TEXT NOT NULL,
    message_count INTEGER,
    file_path TEXT NOT NULL        -- Path to full session JSON in agent's sessions/ dir
);

-- Audit log for all tool executions and self-edits
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    action_string TEXT NOT NULL,   -- tool:<name>:<detail>
    decision TEXT NOT NULL,        -- allow, ask_approved, ask_denied, deny
    user_id INTEGER,              -- Discord user who triggered (NULL for idle-triggered)
    detail TEXT                   -- Additional context (tool params, error messages)
);

-- Persisted messages for rolling context window
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    thread_id INTEGER,
    role TEXT NOT NULL,
    content TEXT,
    tool_calls TEXT,
    tool_call_id TEXT,
    timestamp TEXT NOT NULL,
    discord_message_id INTEGER
);

CREATE INDEX idx_messages_agent_time ON messages(agent_name, timestamp);

-- Execution thread step history (metrics/tracing)
CREATE TABLE thread_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    thread_id INTEGER NOT NULL,
    step_number INTEGER NOT NULL,
    description TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_ms INTEGER
);

CREATE INDEX idx_thread_steps_agent_thread ON thread_steps(agent_name, thread_id);

-- Server-wide default settings
CREATE TABLE settings (
    guild_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (guild_id, key)
);
```

### Filesystem Storage

Agent workspaces live under `~/.chorus-agents/agents/<name>/`. The filesystem stores:

- `agent.json` -- agent configuration (model, system prompt, permissions profile name, channel_id, running_tasks)
- `docs/` -- agent-maintained documentation, always included in context
- `sessions/` -- full session JSON files (message history, tool call logs)
- `workspace/` -- the agent's working directory, git-initialized

Global files under `~/.chorus-agents/`:
- `config.json` -- server-wide defaults (default model, default permissions, etc.)
- `available_models.json` -- cached model discovery results

The split is intentional: SQLite stores metadata that needs to be queried (session listings, audit logs, agent lookups), while the filesystem stores large blobs (full conversation histories, workspace files) that are accessed by path.

---

## Multi-Provider LLM Abstraction

The system supports multiple LLM providers through a protocol-based abstraction. This means the tool loop, context manager, and all other components interact with a `LLMProvider` interface and never import provider-specific SDKs directly.

### Provider Resolution

When an agent is configured with a model like `claude-sonnet-4-20250514` or `gpt-4o`, the system resolves which provider owns that model using the cached `available_models.json`:

```json
{
    "anthropic": {
        "valid": true,
        "models": ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-3-5-20241022"]
    },
    "openai": {
        "valid": true,
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    }
}
```

The provider is selected by finding which provider's model list contains the requested model. If a model appears in multiple providers (unlikely but possible with custom deployments), the first match wins.

### Message Format Normalization

Internally, Chorus uses a common message format:

```python
@dataclass
class Message:
    role: str           # "user", "assistant", "tool_result"
    content: str | None
    tool_calls: list[ToolCall] | None  # For assistant messages with tool_use
    tool_call_id: str | None           # For tool_result messages
```

Each provider implementation translates to/from this common format. The Anthropic provider maps to/from Anthropic's `content_block` format. The OpenAI provider maps to/from OpenAI's `function_call`/`tool_calls` format.

---

## Discord API Limitations

These constraints affect the design of concurrent execution and must be handled explicitly:

- **Rate limits:** 5 messages per 5 seconds per channel. Multiple threads posting output simultaneously will collide. All outgoing messages must go through a per-channel queue with rate limiting.
- **Message length:** 2000 characters. Long LLM responses or tool outputs need truncation, splitting, or file attachments.
- **Interaction timeout:** Slash commands must respond within 3 seconds (can defer for longer work).
- **Embed limits:** 6000 characters total per embed. Status displays must be concise.
- **No persistent UI widgets:** There is no way to keep a live dashboard at the bottom of a channel. Use `/thread list` and `/status` on demand.
- **Reply chains:** Discord's reply feature (`message.reference`) provides free routing metadata. This is the basis of the reply-based thread routing model.

---

## Concurrency Model

### Execution Threads

An agent can run multiple execution threads simultaneously. Each thread is an `asyncio.Task` running an independent tool loop. Threads share the agent's workspace and permission profile but have independent conversation contexts.

```
User sends "refactor auth"              → Thread #1 created, starts tool loop
User sends "write tests for users"      → Thread #2 created, starts tool loop
User replies to Thread #1's message:
  "include error handling too"           → Injected into Thread #1's context
/thread kill 2                          → Thread #2 cancelled
```

### Future: Long-Running Processes (Backlog)

Distinct from execution threads, agents will eventually track OS-level subprocesses that outlive the tool loop (database migrations, build jobs, servers). These are tracked by PID with output captured to log files. See TODO 014.

### Future: Process Hooks (Backlog)

Tracked processes will be able to spawn execution threads on events (exit, output match, timeout). Hook definitions carry context so the spawned thread understands why the process exists and what to do. See TODO 015.

---

## Security Model

Chorus is designed for **private, trusted Discord servers**. It is not safe for public servers.

### Isolation Boundaries

1. **Workspace jail** -- All file operations resolve paths against the agent's workspace directory and verify the resolved path does not escape. This prevents `../../etc/passwd` attacks.
2. **Permission engine** -- Every tool invocation is checked against the agent's permission profile before execution. Denied actions never reach the tool implementation.
3. **Docker container** -- The bot runs inside a Docker container, which provides the outer security boundary for bash execution. There is no chroot, seccomp, or namespace isolation within the container.
4. **Discord roles** -- Self-edit of permissions is gated by the invoking user's Discord role. An agent cannot escalate its own permissions beyond what the user is authorized to grant.

### What is NOT isolated

- Bash commands run as the bot's OS user. An agent with bash access can read/write anything the bot process can.
- Agents on the same host share the same OS. There is no inter-agent isolation beyond separate workspace directories.
- Network access is unrestricted from within bash.

The permission engine is the primary defense against unintended actions. Docker is the hard boundary. For truly untrusted workloads, additional sandboxing (gVisor, Firecracker, etc.) would be needed.
