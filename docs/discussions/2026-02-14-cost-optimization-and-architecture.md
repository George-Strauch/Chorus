# Discussion: Cost Optimization & Architecture Evolution

> **Date:** 2026-02-14
> **Context:** After live testing on Discord, API costs are unsustainably high. This document captures the analysis and decisions around reducing costs and evolving the architecture.

---

## 1. The Problem: Input Token Costs

### Observed Costs (Feb 14, 2026)

| Model | Input Cost | Output Cost | Total |
|-------|-----------|-------------|-------|
| Claude Opus 4.6 | $5.03 | $0.62 | $5.65 |
| Claude Sonnet 4 | $1.10 | $0.11 | $1.21 |
| Claude Haiku 4.5 | $0.00 | $0.01 | $0.01 |
| Claude Haiku 3 | $0.00 | $0.00 | $0.00 |

**Input tokens dominate costs — 89% of total spend.**

### Why Input Tokens Are So High

Every LLM call in the tool loop sends the **full context** fresh:

```
Per LLM call:
├─ System prompt (~37 tokens)
├─ Agent docs (100-1000 tokens)
├─ Model awareness / available models (100-300 tokens)
├─ Scope path info (~150 tokens, if enabled)
├─ Previous branch summary (50-100 tokens)
├─ Thread status (100-1000 tokens)
└─ Conversation history (truncated to ~160K budget)
```

A simple "echo hello" interaction (branch #13: 9,095 in / 60 out) shows ~4,500 tokens of **fixed overhead** before any conversation history. That overhead is repeated on every LLM call in the tool loop.

For a 5-step tool loop, the same ~4,500 tokens of system prompt + docs are sent 5 times = **22,500 tokens just for the static parts**.

### The 200K Cap

We already added a hard 200K input token cap (commit `362954f`) to prevent premium pricing tiers. But within that cap, we're still paying full price for every token because **we are not using prompt caching**.

---

## 2. Missing Optimization: Prompt Caching

### Why Our Billing Shows Zero Cache Tokens

Anthropic's `cache_read_input_tokens` requires explicitly setting `cache_control: {"type": "ephemeral"}` on content blocks. Our `providers.py` does not set these headers — every call pays full input price.

### How Prompt Caching Works

You mark content blocks with `cache_control`. The API caches everything up to that breakpoint. On subsequent calls with the same prefix, cached portions cost 90% less.

```python
# What we should be doing:
system=[
    {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": agent_docs, "cache_control": {"type": "ephemeral"}},
]
```

### Pricing Impact

| Model | Base Input | Cache Write (first call) | Cache Read (subsequent) | Savings |
|-------|-----------|-------------------------|------------------------|---------|
| Opus 4.6 | $5/MTok | $6.25/MTok (1.25x) | **$0.50/MTok (0.1x)** | 90% |
| Sonnet 4 | $3/MTok | $3.75/MTok (1.25x) | **$0.30/MTok (0.1x)** | 90% |
| Haiku 4.5 | $1/MTok | $1.25/MTok (1.25x) | **$0.10/MTok (0.1x)** | 90% |

For a 5-step tool loop with 4,500 static tokens:
- **Without caching:** 4,500 × 5 × $5/MTok = $0.1125
- **With caching:** 4,500 × $6.25/MTok (write) + 4,500 × 4 × $0.50/MTok (reads) = $0.0371
- **Savings: 67% on the static portion alone**

For larger context windows with more docs/history that remain stable across tool loop iterations, savings are even greater.

### Implementation Effort

**Small change to `providers.py`:**
1. Convert system prompt from string to list-of-blocks format
2. Add `cache_control` to the system prompt block and tool definitions
3. Track `cache_creation_input_tokens` and `cache_read_input_tokens` in `Usage`

**Minimum cacheable sizes:**
- Haiku 4.5, Opus 4.5/4.6: 4,096 tokens minimum
- Sonnet 4/4.5, Opus 4/4.1: 1,024 tokens minimum

**TTL:** 5 minutes (default, refreshed on each use) — perfect for tool loops that make multiple calls within seconds.

### Priority: HIGH — Do This First

This is the lowest-effort, highest-impact optimization. It can be implemented in a few hours and immediately reduces costs by 60-90% on the static portions of every LLM call.

---

## 3. Option: Claude Agent SDK Integration

### What Is It?

The **Claude Agent SDK** (`pip install claude-agent-sdk`, v0.1.36) is Anthropic's official library for building agents on top of Claude Code. It provides:

- A complete agentic tool loop (you send a prompt, Claude autonomously reads files, runs commands, edits code)
- Built-in tools: `Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`, `WebSearch`, `WebFetch`
- **Automatic prompt caching** (handled internally)
- Session management (resume, fork, continue)
- Subagent orchestration
- Hooks system (PreToolUse, PostToolUse, Stop)
- Permission control via `can_use_tool` callback
- MCP server integration for custom tools

### Architecture Under the Hood

```
Chorus Bot (Python)
    → claude-agent-sdk (Python package)
    → Claude Code CLI (bundled Node.js binary)
    → stdin/stdout JSON protocol
    → Anthropic API (with automatic caching)
```

### Two Python Interfaces

**`query()` — one-off tasks:**
```python
from claude_agent_sdk import query, ClaudeAgentOptions

async for message in query(
    prompt="Fix the bug in auth.py",
    options=ClaudeAgentOptions(
        allowed_tools=["Read", "Edit", "Bash"],
        cwd=str(agent.workspace_path),
        system_prompt=agent.system_prompt,
        can_use_tool=permission_handler,
    ),
):
    if hasattr(message, "result"):
        await channel.send(message.result)
```

**`ClaudeSDKClient` — persistent multi-turn sessions:**
```python
async with ClaudeSDKClient(options=options) as client:
    await client.query(message_text)
    async for msg in client.receive_response():
        process(msg)
    # Follow-up — Claude remembers context
    await client.query("Now fix the tests")
```

### What Chorus Code It Would Replace

| Chorus Component | Agent SDK Equivalent |
|---|---|
| `tools/file_ops.py` (create, str_replace, view) | Built-in `Read`, `Write`, `Edit` |
| `tools/bash.py` (bash_execute, blocklist) | Built-in `Bash` |
| `tools/git.py` | `Bash` with git commands |
| `tools/registry.py` | Automatic tool management |
| `llm/tool_loop.py` (agentic loop) | `query()` / `ClaudeSDKClient` |
| `llm/providers.py` (Anthropic provider) | Handled internally |
| Manual prompt caching | Automatic prompt caching |

### How Confirmations Would Work

The SDK provides a `can_use_tool` async callback:

```python
async def permission_handler(tool_name, input_data, context):
    action = f"tool:{tool_name}:{summarize(input_data)}"
    result = engine.check(agent.permission_profile, action)

    if result == PermissionResult.ALLOW:
        return PermissionResultAllow(updated_input=input_data)
    elif result == PermissionResult.ASK:
        approved = await discord_ask_ui(channel, tool_name, input_data)
        if approved:
            return PermissionResultAllow(updated_input=input_data)
        return PermissionResultDeny(message="User denied")
    else:
        return PermissionResultDeny(message="Denied by profile")
```

This maps directly to our existing permission engine — the ask UI with Discord buttons would work the same way.

### Plan Mode

```python
options = ClaudeAgentOptions(
    permission_mode="plan",  # Claude will plan but not execute
    # ... other options
)
```

The SDK supports `plan` as a built-in permission mode. Claude will describe what it wants to do and wait for approval before executing.

### Custom Tools via MCP

Chorus-specific tools (self_edit, web_search) can be exposed via in-process MCP servers:

```python
mcp_servers={"chorus": create_sdk_mcp_server(
    name="chorus",
    tools=[self_edit_prompt, self_edit_docs, self_edit_model],
)}
```

### Pros

- Eliminates ~2000 lines of custom tool implementation
- Gets automatic prompt caching for free
- Battle-tested tool implementations
- Session management built-in
- Cost tracking via `ResultMessage.total_cost_usd`
- WebSearch built-in (solves our web search bug)

### Cons

- **Loses OpenAI provider support** — Agent SDK is Claude-only
- Adds Node.js dependency (Claude Code CLI is a bundled Node binary)
- Subprocess layer adds latency
- Less control over exact API parameters (extended thinking, temperature)
- Our path-traversal prevention would be replaced by Claude Code's `cwd` scoping
- More opaque — harder to debug issues in the tool loop
- The Agent SDK is still young (v0.1.x) — API may change

### Assessment

The Agent SDK is a serious option but represents a **major architectural shift**. It replaces the core of Chorus's custom implementation. The trade-off is: less code to maintain, but less control and Claude-only.

**Recommendation:** Don't do this yet. First implement prompt caching (Section 2) and parallel tool execution (TODO 022). If costs are still too high after those optimizations, revisit the Agent SDK as a longer-term migration path.

---

## 4. ChatGPT / OpenAI Considerations

### Current State

Chorus already supports OpenAI models via `OpenAIProvider`. The router uses Haiku or gpt-4o-mini for interjection decisions.

### Cost Comparison

| Model | Input (per MTok) | Output (per MTok) |
|-------|-----------------|-------------------|
| Claude Opus 4.6 | $5.00 | $25.00 |
| Claude Sonnet 4 | $3.00 | $15.00 |
| Claude Haiku 4.5 | $1.00 | $5.00 |
| GPT-4o | $2.50 | $10.00 |
| GPT-4o-mini | $0.15 | $0.60 |
| GPT-4.1 | $2.00 | $8.00 |
| GPT-4.1-mini | $0.40 | $1.60 |
| GPT-4.1-nano | $0.10 | $0.40 |

### Strategy

- **Use cheaper models for routine tasks.** Most agent interactions don't need Opus. Sonnet or GPT-4.1 would handle 90% of tasks.
- **Use Opus/GPT-4o only for complex reasoning.** The agent or user can escalate when needed.
- **Router model should be the cheapest possible.** GPT-4.1-nano at $0.10/MTok input is 10x cheaper than Haiku 4.5.

### OpenAI Prompt Caching

OpenAI also supports prompt caching, but it's **automatic** — no `cache_control` headers needed. Cached reads are 50% off (not 90% like Anthropic). Less savings but zero implementation effort.

### Action Item

Consider adding model-tier awareness: agents default to a cheaper model (Sonnet/GPT-4.1) and only escalate to Opus when the task requires it. This could be a system prompt instruction or a user-configurable setting.

---

## 5. Web Search Bug

### What Happened

The agent enabled web search for itself (via `self_edit_web_search`), but when asked to search, it couldn't. The Discord logs show:

> "I've just enabled web search for myself — but unfortunately it takes effect on the next LLM call"
> "Web search is already enabled but unfortunately I don't have a dedicated web search tool exposed in my current toolset"

### Root Cause

Web search was implemented as an **Anthropic server-side tool** (`web_search_20250305`), not as a custom tool in the registry. The implementation:

1. `self_edit_web_search` toggles `agent.web_search` in `agent.json` ✓
2. When `web_search=True`, the tool spec `{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}` is injected into the tools list ✓
3. Anthropic processes the search server-side and returns results ✓
4. **Bug:** The tool loop tries to look up `web_search` in the local registry → not found → returns `{"error": "Unknown tool: web_search"}` ✗

The server-side tool results need special handling — they don't go through local tool execution. The tool loop doesn't distinguish between "tool calls I need to execute locally" and "tool calls the API already executed server-side."

### What TODO 020 Says vs What Was Implemented

TODO 020 recommended using a third-party search API (Tavily) with a custom local tool. Instead, commit `a2b9ccf` chose Anthropic's server-side tool — simpler (no extra API key) but requires different handling in the tool loop.

### Fix Options

1. **Special-case server-side tools in the tool loop** — skip local execution for `web_search`, pass through the raw result
2. **Register a pass-through handler** — a `web_search` tool in the registry that just returns the server-side result
3. **Switch to the Agent SDK** — has `WebSearch` built-in (but that's a much larger change)
4. **Use a third-party API** — as TODO 020 originally recommended

### Status

TODO 020 should be updated to reflect what was actually implemented and the bug discovered.

---

## 6. Other Behavioral Observations

### Branch Context Isolation Works Correctly

The "echo hello" test (branch #13, 9,095 in / 60 out) shows context scoping working: the agent only saw its own branch history. When asked "what has our conversation been about?" on a new branch, it correctly reported minimal history.

### Context Not Carrying Between Branches (By Design?)

Branch #14 couldn't recall the detailed discussions from branches #10-12 about Claude Code integration and confirmations. This is by design — branches scope context — but the `previous_branch_summary` mechanism should provide a brief summary. The summaries appear too terse ("Echo returns greeting" for branch #13) to be useful for continuity.

**Possible improvement:** Better branch summaries, or a way to reference previous branch content when the user explicitly asks about it.

### Input Token Variance

| Branch | Input Tokens | Output Tokens | Steps | Notes |
|--------|-------------|---------------|-------|-------|
| #8 | 30,573 | 761 | 5 | Web search self-edit (5 tool loop iterations) |
| #9 | 25,304 | 930 | 4 | Web search attempt (4 iterations) |
| #10 | 4,569 | 1,317 | 0 | Simple text response, no tool calls |
| #11 | 66,946 | 1,483 | 7 | Multi-step research (7 iterations) |
| #12 | 70,118 | 1,665 | 10 | Multi-step research (10 iterations) |
| #13 | 9,095 | 60 | 1 | Echo hello (1 iteration, likely tool call) |
| #14 | 4,513 | 97 | 0 | Simple text, no tool calls |
| #14 cont | 4,620 | 13 | 0 | Ping response |

Branch #10 (0 steps, 4,569 in) represents the **minimum baseline**: system prompt + docs + model info + conversation history, no tool calls. This is the fixed overhead per LLM call.

Branch #11 at 66,946 in / 7 steps = ~9,564 per step average. Each step adds the growing conversation to the context.

### The First Message Cost Issue

The user noted that 4,569 tokens for a first message is high. Breaking that down:
- System prompt: ~37 tokens
- Agent docs: varies (likely 500-2000 if the agent has customized its docs)
- Model awareness + available models list: 100-300 tokens
- Scope path info: ~150 tokens
- Thread status: ~100 tokens
- Previous branch summary: ~100 tokens
- Conversation history (user message + any prior): ~200 tokens
- Tool definitions sent with the API call: **~2000-3000 tokens** (all tool schemas)

**The tool schemas are likely the largest hidden cost.** Each tool definition includes name, description, parameter schema. With ~15 tools registered, the schemas alone could be 2000-3000 tokens, sent on every call.

Tool schemas should also be cached (Anthropic caches tools in the same hierarchy as system messages).

---

## 7. Prioritized Action Items

### Immediate (Next Session)

1. **Implement prompt caching in `providers.py`** — add `cache_control` to system prompt and tool definitions for Anthropic calls. Estimated 60-90% reduction in input costs for multi-step tool loops.

2. **Fix the web search bug** — special-case server-side tool results in `tool_loop.py` so they pass through without local execution lookup.

3. **Update TODO 020** — reflect the actual implementation (Anthropic server-side tool) vs the original plan (third-party API), and document the bug/fix.

### Short-Term (This Week)

4. **Implement TODO 022** — parallel tool execution to reduce LLM round-trips (~40% cost reduction on multi-tool workflows).

5. **Add model-tier defaults** — default agents to Sonnet 4 instead of Opus 4.6. Add a system prompt instruction for agents to request model escalation when needed.

### Medium-Term (Evaluate After Above)

6. **Evaluate Claude Agent SDK** — after implementing caching + parallel execution, compare costs. If still too high, consider migrating the tool loop to the Agent SDK for automatic caching and built-in tools.

7. **Consider GPT-4.1/4.1-mini** — for agents that don't need Claude-specific features, OpenAI models may be more cost-effective.

### Not Now

8. **Full Agent SDK migration** — too large a refactor for uncertain benefit. Wait for the SDK to mature (v0.1.x → v1.x) and for clearer cost data after implementing caching.

---

## 8. Decision Log

| Decision | Rationale | Date |
|----------|-----------|------|
| Implement prompt caching first | Lowest effort, highest impact (90% savings on cached tokens) | 2026-02-14 |
| Don't migrate to Agent SDK yet | Too much churn, loses OpenAI support, SDK still v0.1.x | 2026-02-14 |
| Fix web search via tool loop special-casing | Simpler than registering a pass-through, preserves current architecture | 2026-02-14 |
| Create docs/discussions/ for design records | Preserve thought process across sessions | 2026-02-14 |
