# Agent Lifecycle

This document describes the full lifecycle of a Chorus agent, from creation through active use to teardown. It covers the creation process, self-editing capabilities, context management, task tracking, session persistence, and destruction.

---

## Creation

Agents are created via the `/agent init` slash command. The creation process involves multiple steps that set up the agent's isolated workspace, register it with the system, and bind it to a Discord channel.

### Creation Flow

```
/agent init <name> [system_prompt] [model] [permission_profile]
                    |
                    v
1. Validate name
   - Must be unique across all agents (active and destroyed)
   - Must be a valid directory name (alphanumeric, hyphens, underscores)
   - Must be a valid Discord channel name
                    |
                    v
2. Copy template/ -> ~/.chorus-agents/agents/<name>/
   - Recursively copies the template directory from the repo
   - Preserves directory structure: agent.json, docs/, workspace/
                    |
                    v
3. Initialize git repo in workspace/
   - Runs `git init` in ~/.chorus-agents/agents/<name>/workspace/
   - Creates initial commit (empty or with .gitkeep)
                    |
                    v
4. Merge config overrides into agent.json
   - Set "name" field to the agent name
   - If system_prompt provided: override the default
   - If model provided: set it (validated against available_models.json)
   - If model not provided: use server default from config.json
   - If permission_profile provided: set it (validated against known profiles)
   - If permission_profile not provided: use "standard"
   - Set "created_at" to current timestamp
                    |
                    v
5. Register agent in SQLite
   - Insert into agents table: name, channel_id (pending), guild_id, model,
     permissions, created_at, status='active'
                    |
                    v
6. Create Discord channel
   - Create a new text channel in the server named after the agent
   - Store the channel_id back into agent.json and SQLite
   - Optionally set channel topic to system prompt summary
                    |
                    v
7. Send welcome message
   - Post an introductory message in the new channel
   - Include the agent's system prompt and permission profile
   - Agent is now active and listening for messages
```

### Template Contents

The template directory (`template/` in the repo root) contains the minimal scaffolding:

```
template/
  agent.json          # Default config with placeholder values
  docs/
    README.md         # Starter doc for the agent to fill in
  workspace/
    .gitkeep          # Ensures the directory is tracked in git
```

Default `agent.json`:

```json
{
    "name": "",
    "channel_id": null,
    "model": null,
    "system_prompt": "You are a general-purpose AI agent. You have access to a workspace directory where you can create, edit, and view files, run commands, and manage a git repository. Use your tools to accomplish tasks. Maintain notes about your workspace in your docs/ directory.",
    "permissions": "standard",
    "created_at": null,
    "running_tasks": []
}
```

### Post-Creation State

After creation, the agent's directory looks like:

```
~/.chorus-agents/agents/my-agent/
  agent.json              # Populated with name, channel_id, model, timestamps
  docs/
    README.md             # Template starter doc
  sessions/               # Created empty (no saved sessions yet)
  workspace/
    .git/                 # Initialized git repo
    .gitkeep
```

The agent is now active. Messages sent in its Discord channel are routed to the tool loop, and the agent begins processing them using its configured model and tools.

---

## Self-Editing Capabilities

Agents can modify their own configuration through dedicated self-edit tools. This is an intentional first-class capability. The agent's ability to refine its own behavior over time is a core design principle.

### What Agents Can Edit

#### System Prompt

**Tool:** `self_edit_system_prompt(new_prompt: str)`
**Action string:** `tool:self_edit:system_prompt`

The agent can rewrite its own system prompt. This is stored in the `system_prompt` field of `agent.json`. Changes take effect on the next tool loop iteration (the updated prompt is loaded fresh for each LLM call).

Use cases:
- An agent refining its own instructions after learning what works well
- An agent specializing itself for a discovered task pattern
- A user asking the agent to "remember to always do X" and the agent encoding that into its prompt

#### Documentation

**Tool:** `self_edit_docs(path: str, content: str)`
**Action string:** `tool:self_edit:docs:<path>`

The agent can create and edit files in its `docs/` directory. These documents are always included in the agent's context (injected after the system prompt), so changes directly influence future behavior.

This is the primary mechanism for agents to maintain persistent knowledge:
- Project notes, architecture decisions, coding conventions
- Task lists and progress tracking
- Reference information the agent wants to "remember" across sessions

The `docs/` directory is separate from `workspace/` -- docs are private to the agent's own context, while workspace is the actual work product.

Path traversal prevention applies: the path must resolve within the `docs/` directory.

#### Permission Profile

**Tool:** `self_edit_permissions(new_profile: str | dict)`
**Action string:** `tool:self_edit:permissions:<profile_name>`

The agent can change its own permission profile. This is the most sensitive self-edit operation and has an additional gate: the Discord user whose message triggered the current session must have a role that authorizes granting the requested permission level.

See [PERMISSIONS.md](PERMISSIONS.md) for details on the role-based gating mechanism.

The agent can switch to a named preset (`open`, `standard`, `locked`) or provide a custom profile dict. The new profile is validated before being applied:
- Named presets are looked up from the built-in list
- Custom profile dicts have their regex patterns compiled and validated
- Invalid profiles are rejected with an error

#### Model

**Tool:** `self_edit_model(new_model: str)`
**Action string:** `tool:self_edit:model:<model_name>`

The agent can switch its LLM model. The new model is validated against `available_models.json` to ensure a valid API key exists for the corresponding provider.

Use cases:
- Switching to a faster/cheaper model for simple tasks
- Switching to a more capable model for difficult reasoning
- The user asking the agent to "use GPT-4 for this"

The model change takes effect on the next LLM call in the tool loop.

### Self-Edit Audit Trail

All self-edit operations are logged to the SQLite audit table with:
- The agent name
- Timestamp
- The full action string
- The permission decision (allow / ask_approved)
- The Discord user ID who triggered the session
- Detail field containing the old and new values

This makes it possible to trace exactly when and why an agent's configuration changed.

---

## Context Management

Context is the conversation state between a user and an agent. It includes the system prompt, docs contents, message history, and tool call/result pairs.

### Active Session

When a message arrives in an agent's channel:

1. **If no active session exists:** A new session is started. The system prompt and all files in `docs/` are loaded and injected as the context prefix.
2. **If an active session exists:** The new message is appended to the existing conversation history.

The active session is held in memory on the `Agent` object. It is not persisted to disk on every message (that would be too slow). Instead, it is persisted when:
- The user explicitly saves it (`/context save`)
- The idle timer fires
- The bot shuts down gracefully

### Idle Timer

Each agent has an idle timer that tracks inactivity. The timer:
- **Resets** every time the agent processes a message or tool call
- **Fires** after the configured timeout (default: 30 minutes, configurable in server settings)

When the idle timer fires:

```
1. Generate NL summary
   - Send the current conversation history to the LLM with a summarization prompt
   - The summary captures: what was accomplished, what files were changed,
     what decisions were made, what is pending
                    |
                    v
2. Save session
   - Write full message history to sessions/<date>_<summary-slug>.json
   - Insert session metadata into SQLite (id, agent_name, description, saved_at,
     message_count, file_path)
                    |
                    v
3. Clear in-memory context
   - Release the conversation history from memory
   - Agent returns to "no active session" state
                    |
                    v
4. Notify channel
   - Post a message in the Discord channel:
     "Session saved: <summary>. Context cleared after 30 minutes of inactivity.
      Use /context restore <id> to resume."
```

### Manual Save

Users can save the current session at any time with `/context save [description]`:

- If a description is provided, it is used as the session label
- If no description is provided, an NL summary is generated (same as idle timer)
- The session is saved but **context is NOT cleared** -- the agent continues with the same conversation
- This allows creating checkpoints the user can return to

### Manual Clear

`/context clear` immediately clears the agent's active context without saving. The conversation history is discarded. Use this when the context is stale or corrupted and you want a fresh start.

### Context Restore

`/context restore <session_id>` loads a previously saved session:

1. Look up session by ID in SQLite
2. Load the full message history from the session JSON file
3. Replace the agent's current context with the loaded session
4. Notify the channel that context has been restored

The restored session becomes the active session. The agent continues from where that session left off.

### Context History

`/context history [agent_name]` lists all saved sessions for the agent (or a specified agent):

```
Sessions for frontend-agent:
  [abc123] 2026-02-12 10:30 - Refactored auth module, added JWT validation (42 messages)
  [def456] 2026-02-11 15:00 - Initial project setup, configured ESLint (28 messages)
  [ghi789] 2026-02-11 09:00 - Explored codebase, wrote architecture notes (15 messages)
```

### Session File Format

Sessions are stored as JSON files in the agent's `sessions/` directory:

```json
{
    "id": "abc123-uuid-here",
    "agent_name": "frontend-agent",
    "description": "Refactored auth module, added JWT validation",
    "saved_at": "2026-02-12T10:30:00Z",
    "trigger": "idle_timeout",
    "message_count": 42,
    "messages": [
        {
            "role": "user",
            "content": "Let's refactor the auth module to use JWT tokens"
        },
        {
            "role": "assistant",
            "content": "I'll start by examining the current auth implementation.",
            "tool_calls": [
                {
                    "id": "tc_001",
                    "name": "view",
                    "params": {"path": "src/auth.py"}
                }
            ]
        },
        {
            "role": "tool_result",
            "tool_call_id": "tc_001",
            "content": "... file contents ..."
        }
    ]
}
```

The `trigger` field indicates what caused the save: `idle_timeout`, `manual_save`, `shutdown`, or `context_clear`.

---

## Running Tasks

Agents can have long-running operations tracked via the `running_tasks` field in `agent.json`. This is primarily used for background bash commands or multi-step workflows.

### Task Tracking

When an agent starts a long-running bash command (e.g., `npm run build`, `python train.py`), the task is recorded:

```json
{
    "running_tasks": [
        {
            "id": "task_001",
            "command": "npm run build",
            "started_at": "2026-02-12T10:00:00Z",
            "pid": 12345,
            "status": "running"
        }
    ]
}
```

Tasks are updated when they complete (status changes to `completed` or `failed`, exit code is recorded).

### Viewing Tasks

**`/tasks`** -- Lists all running and recently completed tasks for the agent bound to the current channel:

```
Running tasks for frontend-agent:
  [task_001] npm run build       running   (started 5 min ago, PID 12345)
  [task_002] python train.py     running   (started 2 min ago, PID 12346)

Recently completed:
  [task_003] npm test            completed (exit 0, 30 sec ago)
```

**`/status`** -- Shows a comprehensive agent status including tasks, session state, and configuration:

```
Agent: frontend-agent
Model: claude-sonnet-4-20250514
Permissions: standard
Status: active
Session: active (37 messages, idle 5 min)

Running tasks:
  [task_001] npm run build (running, 5 min)

Last session saved: 2026-02-12 09:00 - Initial project setup
```

---

## Teardown

Agents are destroyed via `/agent destroy <name> [--keep-files]`.

### Destruction Flow

```
/agent destroy my-agent
                    |
                    v
1. Save active session (if any)
   - If the agent has an active context, save it as a session
     with trigger="destroy"
                    |
                    v
2. Remove Discord channel
   - Delete the Discord channel associated with the agent
   - This removes all message history from Discord (not from saved sessions)
                    |
                    v
3. Update SQLite
   - Set agent status to 'destroyed' in the agents table
   - Session records and audit logs are preserved
                    |
                    v
4. Handle files
   - Default (no --keep-files): Delete the entire agent directory
     (~/.chorus-agents/agents/<name>/)
   - With --keep-files: Leave the directory intact
     (workspace, sessions, docs, agent.json all preserved)
                    |
                    v
5. Deregister from in-memory agent registry
   - Remove the agent from AgentManager's active agents map
   - The agent no longer receives messages
```

### File Preservation

When `--keep-files` is used, the agent directory remains at `~/.chorus-agents/agents/<name>/`. This is useful when:
- You want to archive the workspace and sessions for later reference
- You plan to recreate the agent with the same name and restore its sessions
- The workspace contains valuable work product (code, documents) you want to keep

The preserved directory is fully self-contained: it includes the agent config, all docs, all saved sessions, and the entire workspace with git history.

### Re-creation After Destruction

An agent destroyed with `--keep-files` can be effectively "restored" by creating a new agent with the same name. However, the creation process will detect the existing directory and handle it:
- The existing `agent.json` is loaded and merged with any new overrides
- The existing workspace (including git history) is preserved
- The existing sessions can be accessed via `/context history` and `/context restore`

Without `--keep-files`, the name becomes available for reuse but all previous state is gone.

---

## State Diagram

```
                    /agent init
                         |
                         v
                   +-----------+
                   |  CREATED  |
                   +-----------+
                         |
                         |  (channel created, listening)
                         v
          +------->+-----------+<--------+
          |        |   IDLE    |         |
          |        +-----------+         |
          |              |               |
          |  idle timer  |  message      |  /context clear
          |  or manual   |  received     |  or idle timeout
          |  restore     v               |
          |        +-----------+         |
          +--------|  ACTIVE   |---------+
                   +-----------+
                         |
                         |  /agent destroy
                         v
                   +-----------+
                   | DESTROYED |
                   +-----------+
```

- **CREATED**: Transient state during initialization. Agent transitions to IDLE immediately after creation is complete.
- **IDLE**: Agent is registered and listening but has no active conversation context. Messages in the channel start a new session.
- **ACTIVE**: Agent has an in-memory conversation context. Messages extend the current session. Idle timer is counting down.
- **DESTROYED**: Agent is deregistered. The Discord channel is deleted. SQLite records are preserved with `status='destroyed'`.

---

## Session Persistence Summary

| Data | Storage Location | Lifetime |
|---|---|---|
| Agent configuration | `agent.json` (filesystem) | Until agent destroyed (or `--keep-files`) |
| Active conversation | In-memory on Agent object | Until saved, cleared, or bot restart |
| Saved sessions (full history) | `sessions/*.json` (filesystem) | Until agent destroyed (or `--keep-files`) |
| Session metadata | SQLite `sessions` table | Permanent (survives destruction) |
| Audit log | SQLite `audit_log` table | Permanent |
| Agent registry | SQLite `agents` table | Permanent (status field tracks lifecycle) |
| Workspace files | `workspace/` (filesystem) | Until agent destroyed (or `--keep-files`) |
| Agent docs | `docs/` (filesystem) | Until agent destroyed (or `--keep-files`) |
| Server defaults | SQLite `settings` table | Permanent |
| Available models cache | `available_models.json` (filesystem) | Until next key validation |

### Graceful Shutdown

When the bot receives a shutdown signal (SIGINT, SIGTERM):

1. All active agent sessions are saved with `trigger="shutdown"`
2. All idle timers are cancelled
3. SQLite connections are closed
4. Discord gateway connection is closed

On restart, agents are loaded from SQLite (status='active'). They start in IDLE state. Users can restore their previous sessions with `/context restore`.
