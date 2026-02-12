# Slash Commands

This document covers every slash command in Chorus, including syntax, parameters, behavior, permission requirements, and examples.

All commands use Discord's slash command system (`app_commands` in discord.py). Commands are registered globally on bot startup.

---

## Command Groups Overview

| Group | Commands | Purpose |
|---|---|---|
| `/agent` | `init`, `destroy`, `list`, `config` | Agent lifecycle management |
| `/settings` | `model`, `permissions`, `show`, `validate-keys` | Server and agent configuration |
| `/context` | `save`, `clear`, `history`, `restore` | Conversation context management |
| `/tasks` | (standalone) | View running tasks for the current agent |
| `/status` | (standalone) | View comprehensive agent status |

---

## /agent

Agent lifecycle commands. These create, destroy, list, and configure agents.

### /agent init

Create a new agent with its own Discord channel and workspace.

**Syntax:**

```
/agent init <name> [system_prompt] [model] [permission_profile]
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | Yes | -- | Agent name. Used as directory name and Discord channel name. Must be alphanumeric with hyphens/underscores. Must be unique. |
| `system_prompt` | string | No | Template default | System prompt for the agent. If omitted, uses the default from `template/agent.json`. |
| `model` | string | No | Server default | LLM model to use (e.g., `claude-sonnet-4-20250514`, `gpt-4o`). Validated against `available_models.json`. If omitted, uses the server default from settings. |
| `permission_profile` | string | No | `standard` | Permission profile name (`open`, `standard`, `locked`) or custom profile. |

**Behavior:**

1. Validates the agent name (unique, valid characters)
2. Copies `template/` to `~/.chorus-agents/agents/<name>/`
3. Initializes git repo in `workspace/`
4. Merges provided overrides into `agent.json`
5. Registers agent in SQLite
6. Creates a new Discord text channel named `<name>`
7. Sends a welcome message in the new channel
8. Returns an ephemeral confirmation to the user

**Permission requirements:** Requires the `Manage Channels` Discord permission (to create the channel). No agent-level permission check (this is a server management operation, not a tool execution).

**Errors:**
- Agent name already exists (active or destroyed without cleanup)
- Invalid characters in name
- Model not found in available models
- Unknown permission profile name
- Missing `Manage Channels` permission

**Examples:**

```
/agent init frontend-agent
/agent init backend-api "You are a backend API specialist. You write Python FastAPI services." claude-sonnet-4-20250514
/agent init docs-writer "You write technical documentation." gpt-4o locked
/agent init devops-agent "You manage CI/CD pipelines and infrastructure." claude-sonnet-4-20250514 open
```

---

### /agent destroy

Destroy an agent, removing its Discord channel and optionally its files.

**Syntax:**

```
/agent destroy <name> [keep_files]
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | Yes | -- | Name of the agent to destroy. Must be an active agent. |
| `keep_files` | boolean | No | `False` | If true, preserve the agent's directory (`~/.chorus-agents/agents/<name>/`). If false, delete it entirely. |

**Behavior:**

1. Saves any active session (with trigger `destroy`)
2. Deletes the Discord channel
3. Updates SQLite status to `destroyed`
4. If `keep_files` is false: deletes the agent directory
5. Removes agent from in-memory registry
6. Returns ephemeral confirmation

**Permission requirements:** Requires the `Manage Channels` Discord permission.

**Errors:**
- Agent not found or already destroyed
- Missing `Manage Channels` permission

**Examples:**

```
/agent destroy frontend-agent
/agent destroy old-project keep_files:True
```

---

### /agent list

List all agents in the server with their status.

**Syntax:**

```
/agent list
```

**Parameters:** None.

**Behavior:**

Returns an ephemeral message listing all agents (active and destroyed) with:
- Agent name
- Status (active / destroyed)
- Bound channel (or "deleted" for destroyed agents)
- Model
- Permission profile
- Creation date
- Active session status (idle / active with message count)

**Permission requirements:** None. Any server member can list agents.

**Output example:**

```
Agents in this server:

  frontend-agent
    Status: active
    Channel: #frontend-agent
    Model: claude-sonnet-4-20250514
    Permissions: standard
    Created: 2026-02-12
    Session: active (23 messages, idle 5 min)

  backend-api
    Status: active
    Channel: #backend-api
    Model: claude-sonnet-4-20250514
    Permissions: open
    Created: 2026-02-11
    Session: idle

  old-experiment (destroyed)
    Destroyed: 2026-02-10
    Files: preserved at ~/.chorus-agents/agents/old-experiment/
```

---

### /agent config

View or update an agent's configuration.

**Syntax:**

```
/agent config <name> <key> [value]
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | Yes | -- | Agent name |
| `key` | string | Yes | -- | Configuration key to view or set. Valid keys: `system_prompt`, `model`, `permissions`, `name` |
| `value` | string | No | -- | New value. If omitted, displays the current value. |

**Behavior:**

- If `value` is provided: updates the specified key in `agent.json`, validates the value, logs the change to the audit table
- If `value` is omitted: displays the current value of the key

This is the operator-level configuration command. Unlike self-edit (which the agent invokes during tool use), `/agent config` is invoked directly by a user and bypasses the permission engine. It is a control plane operation.

**Permission requirements:** Requires the `Manage Channels` Discord permission.

**Configurable keys:**

| Key | Validation | Notes |
|---|---|---|
| `system_prompt` | Any non-empty string | Replaces the agent's system prompt |
| `model` | Must exist in `available_models.json` | Changes the LLM model |
| `permissions` | Must be a known profile name or valid custom JSON | Changes the permission profile |
| `name` | Must be unique, valid characters | Renames the agent (updates directory, channel, SQLite) |

**Examples:**

```
/agent config frontend-agent system_prompt
/agent config frontend-agent model claude-opus-4-20250514
/agent config backend-api permissions open
/agent config docs-writer system_prompt "You are a senior technical writer. Always use active voice."
```

---

## /settings

Server-wide settings management. These affect defaults for new agents and global bot behavior.

### /settings model

Set the default LLM model for new agents.

**Syntax:**

```
/settings model <model_name>
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `model_name` | string | Yes | -- | Model identifier (e.g., `claude-sonnet-4-20250514`, `gpt-4o`). Must be in `available_models.json`. |

**Behavior:**

Sets the server-wide default model in `config.json` and the SQLite settings table. New agents created without an explicit model parameter will use this default.

Does NOT change the model of existing agents. To change an existing agent's model, use `/agent config <name> model <model>`.

**Permission requirements:** Requires the `Administrator` Discord permission.

**Errors:**
- Model not found in available models

**Examples:**

```
/settings model claude-sonnet-4-20250514
/settings model gpt-4o
```

---

### /settings permissions

Set the default permission profile for new agents.

**Syntax:**

```
/settings permissions <profile_name>
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `profile_name` | string | Yes | -- | Profile name: `open`, `standard`, `locked`, or a custom profile name. |

**Behavior:**

Sets the server-wide default permission profile. New agents created without an explicit permission profile will use this default.

Does NOT change existing agents' profiles.

**Permission requirements:** Requires the `Administrator` Discord permission.

**Examples:**

```
/settings permissions standard
/settings permissions locked
```

---

### /settings show

Display all current server-wide settings.

**Syntax:**

```
/settings show
```

**Parameters:** None.

**Behavior:**

Returns an ephemeral message showing all server settings:
- Default model
- Default permission profile
- Idle timeout duration
- Configured API keys (masked, showing only provider name and validity)
- Number of available models per provider
- Permission grant role mappings

**Permission requirements:** Requires the `Administrator` Discord permission.

**Output example:**

```
Server Settings:

  Default model: claude-sonnet-4-20250514
  Default permissions: standard
  Idle timeout: 30 minutes

  API Keys:
    Anthropic: valid (3 models available)
    OpenAI: valid (5 models available)

  Permission grant roles:
    admin -> open, standard, locked
    developer -> standard, locked
```

---

### /settings validate-keys

Re-validate API keys and refresh the available models cache.

**Syntax:**

```
/settings validate-keys
```

**Parameters:** None.

**Behavior:**

1. Tests each configured API key (Anthropic, OpenAI) by making a validation request
2. For valid keys, queries the provider for available models
3. Updates `~/.chorus-agents/available_models.json`
4. Returns a summary of results

This is useful after adding new API keys to `.env` and restarting, or when new models become available from a provider.

**Permission requirements:** Requires the `Administrator` Discord permission.

**Output example:**

```
Key Validation Results:

  Anthropic: VALID
    Models: claude-opus-4-20250514, claude-sonnet-4-20250514, claude-haiku-3-5-20241022

  OpenAI: VALID
    Models: gpt-4o, gpt-4o-mini, gpt-4-turbo

  Available models cache updated.
```

---

## /context

Context management commands for saving, clearing, and restoring conversation sessions.

These commands operate on the agent bound to the **current channel**. If invoked in a channel that is not bound to an agent, an error is returned.

### /context save

Save the current conversation session.

**Syntax:**

```
/context save [description]
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `description` | string | No | Auto-generated NL summary | A label for this session save point. If omitted, the LLM generates a natural-language summary. |

**Behavior:**

1. If no active session exists, returns an error ("No active session to save")
2. If description is omitted, calls the LLM to generate a summary of the conversation
3. Writes the full message history to `sessions/<date>_<slug>.json`
4. Inserts session metadata into SQLite
5. Returns the session ID and description
6. The active session continues (context is NOT cleared)

This creates a checkpoint. The agent continues with the same context. You can save multiple times during a long session.

**Permission requirements:** None. Any user can save context for the agent in their channel.

**Output example:**

```
Session saved.
  ID: abc12345
  Description: Refactored auth module, added JWT validation middleware
  Messages: 42
  Use /context restore abc12345 to return to this point.
```

**Examples:**

```
/context save
/context save "Before attempting the database migration"
/context save "Checkpoint after fixing all lint errors"
```

---

### /context clear

Clear the agent's active conversation context.

**Syntax:**

```
/context clear
```

**Parameters:** None.

**Behavior:**

1. If no active session exists, returns a message ("No active session to clear")
2. Discards the in-memory conversation history without saving
3. Cancels the idle timer
4. Agent returns to IDLE state
5. Returns confirmation

The context is gone. If you want to save before clearing, use `/context save` first.

**Permission requirements:** None. Any user in the channel can clear context.

**Output example:**

```
Context cleared for frontend-agent. 37 messages discarded.
The agent will start fresh on the next message.
```

---

### /context history

List saved sessions for an agent.

**Syntax:**

```
/context history [agent_name]
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `agent_name` | string | No | Current channel's agent | Agent to list sessions for. If omitted, uses the agent bound to the current channel. |

**Behavior:**

Queries SQLite for all sessions belonging to the specified agent, ordered by saved_at descending. Returns a formatted list with session IDs, dates, descriptions, and message counts.

**Permission requirements:** None. Any server member can view session history.

**Output example:**

```
Sessions for frontend-agent:

  [abc12345] 2026-02-12 10:30
    Refactored auth module, added JWT validation middleware
    42 messages | saved by idle timeout

  [def67890] 2026-02-11 15:00
    Initial project setup, configured ESLint and Prettier
    28 messages | saved manually

  [ghi11111] 2026-02-11 09:00
    Explored codebase, wrote architecture decision records
    15 messages | saved on shutdown
```

**Examples:**

```
/context history
/context history backend-api
```

---

### /context restore

Restore a previously saved session as the active context.

**Syntax:**

```
/context restore <session_id>
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `session_id` | string | Yes | -- | The session ID to restore (from `/context history` output). |

**Behavior:**

1. Looks up the session in SQLite by ID
2. Verifies the session belongs to the agent bound to the current channel
3. Loads the full message history from the session JSON file
4. If the agent has an active session, it is replaced (discarded without saving)
5. The restored session becomes the active context
6. Resets the idle timer
7. Returns confirmation with session details

After restoration, the agent continues as if the conversation never ended. The next message from the user is appended to the restored history.

**Permission requirements:** None. Any user in the channel can restore context.

**Warning:** Restoring a session discards any current active context without saving it. Use `/context save` first if you want to preserve the current session.

**Output example:**

```
Session restored for frontend-agent.
  ID: abc12345
  Description: Refactored auth module, added JWT validation middleware
  Messages: 42
  Originally saved: 2026-02-12 10:30

The agent will continue from this session.
```

**Examples:**

```
/context restore abc12345
/context restore def67890
```

---

## /tasks

View running and recently completed tasks for the agent in the current channel.

**Syntax:**

```
/tasks
```

**Parameters:** None.

**Behavior:**

1. Identifies the agent bound to the current channel
2. Reads the `running_tasks` list from `agent.json`
3. Checks the status of each tracked process (running, completed, failed)
4. Returns a formatted list of tasks with their status, duration, and PID

Tasks are bash commands that the agent started during tool use. They are tracked when the bash tool launches a long-running process.

**Permission requirements:** None. Any user in the channel can view tasks.

**Output example:**

```
Tasks for frontend-agent:

  Running:
    [task_001] npm run build
      PID: 12345 | started 5 min ago

    [task_002] python scripts/train_model.py
      PID: 12346 | started 2 min ago

  Recently completed:
    [task_003] npm test
      Exit code: 0 | completed 30 sec ago | duration: 45 sec

    [task_004] eslint src/
      Exit code: 1 | completed 5 min ago | duration: 12 sec
```

If no tasks exist:

```
No tasks for frontend-agent.
```

---

## /status

Show comprehensive status for the agent in the current channel.

**Syntax:**

```
/status
```

**Parameters:** None.

**Behavior:**

Returns a detailed status view of the agent including:
- Agent name and creation date
- Current model
- Current permission profile
- Active session state (idle, active with message count and idle duration)
- Running tasks summary
- Last saved session info
- Workspace git status (current branch, uncommitted changes count)

**Permission requirements:** None. Any user in the channel can view status.

**Output example:**

```
Agent: frontend-agent
Created: 2026-02-12
Model: claude-sonnet-4-20250514
Permissions: standard

Session: active
  Messages: 37
  Idle: 5 minutes
  Idle timeout: 25 minutes remaining

Tasks: 2 running, 1 completed recently
  [task_001] npm run build (running, 5 min)
  [task_002] python train.py (running, 2 min)

Last saved session:
  [abc12345] 2026-02-12 09:00 - Initial project setup (28 messages)

Workspace:
  Branch: feature/auth-refactor
  Uncommitted changes: 3 files
```

---

## Error Handling

All commands follow consistent error handling:

1. **Ephemeral responses** -- Error messages are sent as ephemeral responses (only visible to the invoking user), not posted in the channel for everyone to see.

2. **Validation errors** include the specific issue:
   ```
   Error: Model "gpt-5" not found in available models.
   Available models: claude-sonnet-4-20250514, claude-opus-4-20250514, gpt-4o, gpt-4o-mini
   ```

3. **Permission errors** reference the required Discord permission:
   ```
   Error: This command requires the Manage Channels permission.
   ```

4. **Channel binding errors** explain what happened:
   ```
   Error: This channel is not bound to an agent.
   Use /agent init to create an agent, or run this command in an agent's channel.
   ```

---

## Command Registration

Commands are registered in `bot.py` using discord.py's `app_commands` system:

```python
# Agent commands (group)
agent_group = app_commands.Group(name="agent", description="Agent management")

@agent_group.command(name="init")
async def agent_init(interaction, name: str, system_prompt: str = None, ...):
    ...

# Settings commands (group)
settings_group = app_commands.Group(name="settings", description="Server settings")

# Context commands (group)
context_group = app_commands.Group(name="context", description="Context management")

# Standalone commands
@bot.tree.command(name="tasks")
async def tasks(interaction):
    ...

@bot.tree.command(name="status")
async def status(interaction):
    ...
```

Command implementations live in their respective modules under `commands/`:
- `commands/agent_commands.py` -- `/agent` subcommands
- `commands/settings_commands.py` -- `/settings` subcommands
- `commands/context_commands.py` -- `/context` subcommands

The standalone `/tasks` and `/status` commands are lightweight and may be defined directly in `bot.py` or in a separate `commands/status_commands.py` module.
