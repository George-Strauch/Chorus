# Security

Chorus gives LLM agents direct OS-level access — shell, filesystem, and git — running as the bot's OS user. This document describes the threat model, the safety mechanisms in place, and the known limitations.

## Threat Model

Chorus is designed for **private, trusted Discord servers only**. The architecture deliberately trades security isolation for convenience and capability. The core assumptions:

- **Every server member is trusted.** Any member who can see an agent channel can instruct that agent to do anything its permissions allow.
- **The permission engine is a safety net, not a security boundary.** It reduces risk from careless LLM actions but cannot prevent a determined or confused agent from causing damage if misconfigured.
- **Docker is the only hard security boundary.** Without Docker, agents run with full access as the bot's OS user.
- **LLMs are unpredictable.** Agents may misunderstand instructions, hallucinate commands, or take actions you didn't intend. Safety mechanisms exist to catch common mistakes, not to guarantee correctness.

## Safety Mechanisms

### 1. Permission Engine (Allow / Ask / Deny)

Every tool action is represented as a string (`tool:<name>:<detail>`) and matched against the agent's permission profile using regex patterns.

- **Allow** — action proceeds automatically
- **Ask** — action requires explicit human approval via Discord buttons
- **Deny** — action is blocked (default for anything not matching allow or ask)

Built-in presets: `open` (allow all), `standard` (file ops auto, bash/git ask), `locked` (view-only).

### 2. Path Traversal Prevention

All file operations resolve paths and verify containment before any I/O:

- Symlinks resolved before the check (no symlink escape)
- Canonical path must be inside the agent's workspace root
- Raises `PathTraversalError` on escape attempt

### 3. Bash Command Blocklist

A best-effort blocklist catches catastrophic command patterns before execution:

- `rm -rf /` variants
- Fork bombs (`:() { :|:& };:`)
- Disk fill (`dd if=/dev/zero`)
- Filesystem format (`mkfs`)
- Device overwrite (`> /dev/sd*`)

### 4. Environment Sanitization

Bash subprocesses receive a minimal environment — only 9 variables are passed through:

`PATH`, `HOME`, `USER`, `LANG`, `LC_ALL`, `TERM`, `SHELL`, `TMPDIR`, `SCOPE_PATH`

`HOME` is jailed to the agent's workspace directory. All other environment variables (including API keys) are stripped.

### 5. Ask / Approve UI

Actions matching `ask` patterns present a Discord button prompt to the user who triggered the action. The prompt shows the tool name, arguments, and formatted action string. Timeout is 120 seconds — expiry results in denial.

### 6. Audit Logging

All tool executions are logged to SQLite with the action string, permission decision (allow/ask/deny), timestamp, and agent name.

### 7. Agent Name Validation

Agent names are validated against a strict regex pattern to prevent path injection and other naming attacks.

## Known Limitations

- **Blocklist is circumventable.** It catches common patterns but cannot block all destructive commands. Creative shell tricks, encoded commands, or multi-step approaches can bypass it.
- **No user isolation.** All agents run as the same OS user. There is no per-agent Linux user, namespace, or cgroup boundary.
- **Bash allows absolute paths.** While file operations are jailed to the workspace, bash commands can reference any path the OS user can access.
- **Docker socket escape.** If the Docker socket is mounted (for agents that manage containers), it provides effective root access to the host.
- **LLM unpredictability.** No safety mechanism can fully account for the fact that LLMs sometimes do things you didn't ask for, misunderstand context, or confidently execute incorrect commands.
- **Self-edit scope.** Agents can modify their own system prompt and docs. A confused agent could weaken its own safety instructions. Permission profile self-edits require the invoking user to have Discord `manage_guild` permission.

## Docker Security

When running under Docker:

- The bot runs as an unprivileged user inside the container
- `SCOPE_PATH` limits the agent workspace root
- Docker socket mount is optional — omit it unless agents need container management
- The container is the blast radius boundary for filesystem and process damage

## Reporting Vulnerabilities

If you find a security issue, please open an issue on the [GitHub repository](https://github.com/georgestrauch/Chorus/issues). For sensitive reports, include `[SECURITY]` in the title.
