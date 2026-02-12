# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chorus is a Python Discord bot that turns Discord channels into autonomous AI agents. Each channel maps to a dedicated agent with its own working directory, permission profile, system prompt, tools, and git repo. Agents can edit files, run commands, create merge requests, manage their own context, and **edit their own configuration** — all orchestrated through Discord slash commands.

Agents are **general purpose**. They can code, write, research, manage projects, maintain documentation, orchestrate workflows — anything an LLM with file/bash/git tools can do. The architecture must reflect this: no assumptions about "coding agent" vs "writing agent." An agent is a workspace + tools + permissions + a system prompt. What it does is defined by how you configure it.

**Key principle:** Discord IS the interface. Channels are workspaces. Roles are permissions. Slash commands are the control plane.

---

## Build & Test Commands

```bash
# Activate venv
source .venv/bin/activate

# Install dependencies (editable + dev)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_config.py

# Run a single test
pytest tests/test_config.py::test_function_name -v

# Type checking
mypy src/chorus/

# Linting
ruff check src/ tests/
ruff format src/ tests/

# Run the bot
python -m chorus
```

---

## Architecture

```
chorus/                          # Project root (this repo)
├── CLAUDE.md                    # This file
├── README.md
├── pyproject.toml               # Project config (use modern Python packaging)
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── .env.example                 # Template for secrets (Anthropic + OpenAI keys)
├── docs/
│   ├── ARCHITECTURE.md          # System design doc
│   ├── PERMISSIONS.md           # Permission profile spec (allow/ask regex model)
│   ├── AGENT_LIFECYCLE.md       # Agent init (from template), self-edit, context save, teardown
│   ├── COMMANDS.md              # All slash commands documented
│   └── todo/
│       ├── 001-core-bot.md              # Discord bot skeleton + slash command registration
│       ├── 002-agent-manager.md         # Agent creation from template, directory setup, lifecycle
│       ├── 003-permission-profiles.md   # Regex-based allow/ask permission system
│       ├── 004-file-tools.md            # File editing tools (create, edit, view, str_replace)
│       ├── 005-bash-execution.md        # Sandboxed command execution per agent
│       ├── 006-git-integration.md       # Git operations, branch management, MR creation
│       ├── 007-context-management.md    # Session save/clear, idle timer, NL summaries
│       ├── 008-llm-integration.md       # Multi-provider LLM client, key validation, model discovery
│       ├── 009-default-settings.md      # Server-wide defaults (model, permissions, etc.)
│       ├── 010-agent-self-edit.md       # Agents editing their own config, prompt, tools
│       ├── 011-voice-agent.md           # Voice channel agent (future)
│       ├── 012-webhooks.md              # Webhook system for external actions (future)
│       └── 013-agent-communication.md   # Inter-agent messaging (future)
├── template/                    # Template agent — copied when creating new agents
│   ├── agent.json               # Default agent config (model, system prompt, permissions)
│   ├── docs/                    # Agent-local documentation the agent can read and edit
│   │   └── README.md            # Describes this agent's purpose (agent fills this in)
│   └── workspace/               # Empty workspace, git-initialized on creation
│       └── .gitkeep
├── src/
│   └── chorus/
│       ├── __init__.py
│       ├── __main__.py          # Entry point for python -m chorus
│       ├── bot.py               # Discord bot entrypoint, slash command registration
│       ├── config.py            # Bot config, env vars, defaults management
│       ├── models.py            # Data models (Agent, PermissionProfile, Session, etc.)
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── manager.py       # Agent lifecycle: create (from template), destroy, list, configure
│       │   ├── directory.py     # Agent working directory management (~/.chorus-agents/)
│       │   ├── context.py       # Context window management, session save/restore
│       │   └── self_edit.py     # Agent self-modification (config, prompt, docs, permissions)
│       ├── permissions/
│       │   ├── __init__.py
│       │   └── engine.py        # Regex-based allow/ask permission engine
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── registry.py      # Tool registry — all tools registered here, filtered by permissions
│       │   ├── file_ops.py      # create_file, str_replace, view within agent workspace
│       │   ├── bash.py          # Sandboxed bash execution within agent directory
│       │   └── git.py           # Git operations: init, commit, push, branch, merge request
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── providers.py     # Multi-provider client (Anthropic, OpenAI), key validation
│       │   ├── discovery.py     # Test keys, discover available models per provider
│       │   └── tool_loop.py     # Agentic tool use loop (call → tool → call → ...)
│       ├── commands/
│       │   ├── __init__.py
│       │   ├── agent_commands.py    # /agent init, /agent destroy, /agent list, /agent config
│       │   ├── settings_commands.py # /settings model, /settings permissions, /settings defaults
│       │   └── context_commands.py  # /context save, /context clear, /context history
│       └── storage/
│           ├── __init__.py
│           └── db.py            # SQLite storage for sessions, agent state, settings, audit log
└── tests/
    ├── __init__.py
    ├── conftest.py              # Shared fixtures (mock Discord ctx, mock agent dirs, tmp template, etc.)
    ├── test_config.py
    ├── test_models.py
    ├── test_agent/
    │   ├── __init__.py
    │   ├── test_manager.py
    │   ├── test_directory.py
    │   ├── test_context.py
    │   └── test_self_edit.py
    ├── test_permissions/
    │   ├── __init__.py
    │   └── test_engine.py
    ├── test_tools/
    │   ├── __init__.py
    │   ├── test_registry.py
    │   ├── test_file_ops.py
    │   ├── test_bash.py
    │   └── test_git.py
    ├── test_llm/
    │   ├── __init__.py
    │   ├── test_providers.py
    │   ├── test_discovery.py
    │   └── test_tool_loop.py
    ├── test_commands/
    │   ├── __init__.py
    │   ├── test_agent_commands.py
    │   ├── test_settings_commands.py
    │   └── test_context_commands.py
    └── test_storage/
        ├── __init__.py
        └── test_db.py
```

### Agent Directory Structure (~/.chorus-agents/)

Each agent gets its own isolated workspace, created by copying `template/`:

```
~/.chorus-agents/
├── config.json                  # Global defaults (default model, default permissions, etc.)
├── available_models.json        # Cached result of key validation / model discovery
├── agents/
│   ├── frontend-agent/          # Copied from template/, then customized
│   │   ├── agent.json           # Agent config (model, system prompt, permissions, channel_id)
│   │   ├── docs/                # Agent-local docs — the agent can read AND edit these
│   │   │   └── README.md
│   │   ├── sessions/            # Saved context sessions with NL descriptions
│   │   │   ├── 2026-02-12_auth-refactor.json
│   │   │   └── 2026-02-11_initial-setup.json
│   │   └── workspace/           # The agent's actual working directory
│   │       ├── .git/
│   │       └── ... (project files)
│   └── writing-agent/           # Same structure — but this one writes prose, not code
│       ├── agent.json
│       ├── docs/
│       ├── sessions/
│       └── workspace/
└── db/
    └── chorus.db                # SQLite: sessions, audit log, permission grants
```

---

## Core Concepts

### Template Agent

The `template/` directory in the repo root is the blueprint for all new agents. When `/agent init <name>` is called:

1. Copy `template/` → `~/.chorus-agents/agents/<name>/`
2. Initialize a git repo in `workspace/`
3. Merge any provided overrides (model, system prompt, permissions) into `agent.json`
4. Register the agent in SQLite
5. Create the Discord channel

The template ships with sensible defaults but is minimal. The agent can then edit its own docs, config, and even its system prompt as it works.

### Agent Self-Editing

Agents can modify their own configuration. This is a first-class capability, not a hack. The self-edit tools allow an agent to:

- **Edit its own system prompt** (stored in `agent.json`)
- **Edit its own docs** (in its `docs/` directory)
- **Adjust its own permission profile** — only if the *user's* Discord role allows granting that permission level
- **Update its model** — switch between available models as needed

Self-edits are logged in the audit table. The agent's `docs/` directory is always included in its context, so edits there directly influence future behavior.

### Permission Profiles

Permissions use a **regex-based allow/ask model**. Every action is represented as a string and matched against the profile:

```python
@dataclass
class PermissionProfile:
    allow: list[str]  # Regex patterns — matched actions proceed automatically
    ask: list[str]    # Regex patterns — matched actions require human confirmation
```

Action string format: `tool:<tool_name>:<detail>` (e.g., `tool:bash:rm -rf /`, `tool:git:push origin main`, `tool:self_edit:system_prompt`)

Matching order: allow → ask → DENY

**Built-in presets:** `open` (allow all), `standard` (file ops auto, bash/push ask), `locked` (view-only)

### Slash Commands

```
/agent init <name> [system_prompt] [model] [permission_profile]
/agent destroy <name> [--keep-files]
/agent list
/agent config <name> <key> <value>
/settings model <model_name>
/settings permissions <profile_name>
/settings show
/settings validate-keys
/context save [description]
/context clear
/context history [agent_name]
/context restore <session_id>
/tasks                          # View agent's running programs
/status                         # Agent status including running tasks
```

### Multi-Provider LLM Support

Supports Anthropic and OpenAI. Keys validated on startup, available models discovered and cached in `~/.chorus-agents/available_models.json`. Providers implement a unified `LLMProvider` protocol so the tool loop is provider-agnostic.

### Context Management & Idle Timer

- Idle timeout (default 30min) triggers: NL summary → save session → clear context → notify channel
- Manual save/restore via `/context save` and `/context restore`
- Sessions stored in agent's `sessions/` directory + metadata in SQLite

---

## Tech Stack

- **Python 3.12+** with `pathlib.Path` everywhere
- **discord.py** (slash commands via app_commands)
- **anthropic** + **openai** (official SDKs)
- **aiosqlite** (async SQLite)
- **pytest + pytest-asyncio** (TDD)
- **mypy** (strict type checking)
- **ruff** (linting + formatting)
- **Docker** (deployment)

---

## Implementation Order

Follow `docs/todo/` in numerical order. Each TODO: objective, acceptance criteria, tests-first, implementation notes, dependencies.

**Phase 1 — Foundation (001-003):** Bot skeleton, agent manager, permission engine
**Phase 2 — Agent Tools (004-006):** File ops, bash execution, git integration
**Phase 3 — Intelligence (007-010):** Context management, LLM integration, defaults, self-edit
**Phase 4 — Future (011-013):** Voice, webhooks, inter-agent communication

---

## Conventions

- **Agents are general purpose.** Never assume a task domain.
- All paths within agent tools are **relative to the agent's workspace**. Path traversal prevention is mandatory (canonicalize + jail check).
- All Discord interactions are async.
- All tool executions logged to SQLite audit table with action string + allow/ask/deny result.
- Environment variables in `.env`, never committed. `.env.example` shows the shape.
- Type hints on everything. Permission action strings: `tool:<tool_name>:<detail>`.
- **TDD**: write tests first, run them (fail), implement, pass, refactor.

---

## Git Branching Strategy

Two main branches: **`master`** and **`automate`**.

- **`master`** is write-protected. Never commit directly to master. Changes reach master only via merge from `automate`, and we tag master after each merge.
- **`automate`** is the working integration branch. Most changes land here.
- **Feature/change branches** are checked out from `automate` (e.g. `feat/001-core-bot`, `fix/ping-latency`). When the work is done, merge the feature branch into `automate` and delete it.

**Before starting any new work:**
1. Check what branch you are on (`git branch --show-current`).
2. If you are on a stale feature branch with unmerged changes, ask the user whether to merge it into `automate` before proceeding.
3. Create a new feature branch from `automate` for the new work.

---

## Security

**This is a dangerous repo.** Chorus gives LLM agents unrestricted OS-level access (shell, filesystem, git) as the bot's user. The threat surface is large from both a security perspective and from the risk of reckless or incompetent LLM behavior. This trade-off is accepted deliberately for convenience and utility.

- **Private, trusted servers only.** Only run on servers with people you would trust to work on your computer unsupervised. README includes a prominent warning — keep it there and keep it accurate.
- Bash runs as bot's OS user, scoped to agent workspace via `cwd`. Docker is the only hard security boundary.
- The permission engine is a safety net, not a security boundary. It reduces risk from careless LLM actions but does not prevent a determined or confused agent from causing damage if misconfigured.
- Path traversal prevention in file_ops, bash, and self_edit — resolve and check before any I/O.
- Bot token and API keys never logged or exposed in Discord messages.
- Permission checks happen BEFORE tool execution.
- Agent self-edit of permissions gated by invoking user's Discord role.
