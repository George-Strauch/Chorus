# Backlog: Claude Code SDK / Agent SDK Integration

> **Status:** IDEA — evaluate after implementing prompt caching (TODO 023) and parallel execution (TODO 022)
> **Added:** 2026-02-14

## Overview

The [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) (`pip install claude-agent-sdk`, v0.1.36) is Anthropic's official Python library for building agents on top of Claude Code. It could replace Chorus's entire custom tool loop, tool implementations, and provider layer.

## What It Provides

- **Built-in tools:** Read, Write, Edit, Bash, Glob, Grep, WebSearch, WebFetch, NotebookEdit
- **Automatic prompt caching** — handled transparently, no `cache_control` headers needed
- **Agentic tool loop** — `query()` for one-off, `ClaudeSDKClient` for persistent sessions
- **Session management** — resume, fork, continue conversations
- **Subagent orchestration** — spawn specialized sub-agents
- **Hooks system** — PreToolUse, PostToolUse, Stop, UserPromptSubmit
- **Permission control** — `can_use_tool` async callback (maps to our permission engine)
- **MCP server support** — expose custom tools (self_edit, etc.) via in-process MCP
- **Cost tracking** — `ResultMessage.total_cost_usd`
- **Plan mode** — `permission_mode="plan"` built-in

## Architecture

```
Chorus Bot (Python)
    → claude-agent-sdk (Python package)
    → Claude Code CLI (bundled Node.js binary)
    → stdin/stdout JSON protocol
    → Anthropic API (with automatic caching)
```

## What It Would Replace

| Chorus Component | Agent SDK Equivalent |
|---|---|
| `tools/file_ops.py` (~300 lines) | Built-in Read, Write, Edit |
| `tools/bash.py` (~200 lines) | Built-in Bash |
| `tools/git.py` (~150 lines) | Bash with git commands |
| `tools/registry.py` (~500 lines) | Automatic tool management |
| `llm/tool_loop.py` (~600 lines) | `query()` / `ClaudeSDKClient` |
| `llm/providers.py` (~300 lines) | Handled internally |
| Total: ~2,050 lines | Replaced by SDK dependency |

## How Permissions Would Work

The SDK provides a `can_use_tool` async callback that maps directly to our permission engine:

```python
async def permission_handler(tool_name, input_data, context):
    action = f"tool:{tool_name}:{summarize(input_data)}"
    result = engine.check(agent.permission_profile, action)

    if result == PermissionResult.ALLOW:
        return PermissionResultAllow(updated_input=input_data)
    elif result == PermissionResult.ASK:
        approved = await discord_ask_ui(channel, tool_name, input_data)
        return PermissionResultAllow() if approved else PermissionResultDeny()
    else:
        return PermissionResultDeny(message="Denied by profile")
```

## Pros

- Eliminates ~2,000 lines of custom tool code
- Automatic prompt caching (biggest cost saver)
- Battle-tested, optimized tool implementations
- Session management built-in
- WebSearch built-in (solves our web search bug for free)
- Plan mode built-in
- Cost tracking per-turn

## Cons

- **Drops OpenAI provider support** — Agent SDK is Claude-only
- **Node.js dependency** — Claude Code CLI is a bundled Node binary
- **Subprocess layer** — Python → Node → API adds latency
- **Less control** — can't fine-tune API parameters (extended thinking, temperature)
- **Opaque debugging** — harder to trace issues through the SDK/CLI/API chain
- **SDK is v0.1.x** — API may change, breaking our integration
- **Workspace scoping** — Claude Code uses `cwd`, not our custom path traversal checks
- **No multi-provider** — can't use GPT-4.1 for cheap tasks, Claude for complex ones

## Decision

**Not now.** The SDK is promising but the trade-offs are significant:
1. Losing OpenAI support limits our model tiering strategy
2. The SDK is too young (v0.1.x) for production dependency
3. Implementing prompt caching ourselves (TODO 023) gets 80% of the cost benefit
4. Our custom tool implementations give us fine-grained control

**Revisit when:**
- The Agent SDK reaches v1.0
- We've implemented prompt caching and parallel execution and costs are still too high
- OpenAI support is added to the SDK
- We want plan mode (TODO 018) and don't want to build it ourselves

## Resources

- [Agent SDK Python docs](https://platform.claude.com/docs/en/agent-sdk/python)
- [Agent SDK GitHub](https://github.com/anthropics/claude-agent-sdk-python)
- [Agent SDK Demos](https://github.com/anthropics/claude-agent-sdk-demos)
- [Headless Mode (CLI invocation)](https://code.claude.com/docs/en/headless)
- [Prompt Caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- [Discussion: 2026-02-14 Cost Optimization](../discussions/2026-02-14-cost-optimization-and-architecture.md)
