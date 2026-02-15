# TODO 028 — Claude Code SDK Integration

> **Status:** COMPLETE
> **Branch:** `feat/028-claude-code-integration`
> **Depends on:** 001–010 (core platform), 023 (prompt caching)

## Objective

Add `claude_code` as a new tool that delegates coding tasks to Claude Code via the `claude-agent-sdk`. The bot's LLM decides when to invoke it (guided by system prompt). Each invocation is a one-shot `query()` call — the bot passes a task description, Claude Code does the work, results come back.

## Key Decisions

- **Tool, not replacement**: `claude_code` is a new tool alongside existing file/bash tools, not a replacement
- **`query()` not `ClaudeSDKClient`**: Each invocation is stateless; the outer Chorus tool loop provides continuity
- **Conditional registration**: Tool absent (not broken) when SDK not installed
- **`permission_mode="acceptEdits"`**: Claude Code freely edits within workspace; Chorus checks `tool:claude_code:*` permission before invoking
- **Host execution opt-in** (`CHORUS_HOST_EXEC=false` default): Safe by default
- **Output truncated at 50KB**: Prevent large output from blowing up context window
- **Model forwarding**: Pass agent's model to Claude Code when it's an Anthropic model

## Acceptance Criteria

1. `claude_code` tool registered when `claude-agent-sdk` is installed
2. Tool absent (not broken) when SDK not installed — graceful degradation
3. `tool:claude_code:<task>` action strings checked by permission engine
4. Standard preset: ASK for claude_code; open: ALLOW; locked: DENY
5. Successful execution returns output, cost, duration, turn count
6. SDK errors (not installed, timeout, process error) handled gracefully
7. Output truncated at 50KB
8. `host_execution` flag controls bash environment (full env vs sanitized)
9. System prompt includes Claude Code awareness when SDK available
10. All existing tests continue to pass

## Tests First

- `TestClaudeCodeResult` — serialization
- `TestIsClaudeCodeAvailable` — import check (mock import failure)
- `TestClaudeCodeExecute` — mock `claude_agent_sdk.query()`: success, SDK missing, timeout, CLI not found, process error, output truncation
- `TestClaudeCodeRegistration` — tool registered when SDK available, not when absent
- `TestClaudeCodePermissions` — action string built correctly, presets match
- `TestHostExecution` — bash respects host_execution flag

## Implementation Notes

- SDK invocation uses `query()` with `ClaudeAgentOptions(cwd=workspace, permission_mode="acceptEdits")`
- When `host_execution=True`: bash passes through full environment, doesn't jail HOME
- When `host_execution=False` (default): bash uses sanitized env as before
- `CHORUS_HOST_EXEC` env var controls the flag
- Dockerfile adds Node.js 18 (required runtime for Claude Code CLI)
