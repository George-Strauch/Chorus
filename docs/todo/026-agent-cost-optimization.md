# TODO 026 — Agent Cost Optimization

> **Status:** PENDING
> **Priority:** Medium — reduces ongoing operational costs
> **Dependencies:** 023 (prompt caching), 009 (default settings)

## Objective

Reduce per-interaction API costs through model tiering, context size reduction, and smarter defaults. This complements prompt caching (TODO 023) and parallel tool execution (TODO 022) with higher-level optimizations.

## Current Cost Breakdown

A typical agent interaction on Opus 4.6:
- Base overhead per LLM call: ~4,500 tokens (system prompt + docs + tool schemas)
- Tool schemas alone: ~2,500 tokens (15 tools × ~170 tokens each)
- 5-step tool loop: ~$0.50-$2.00 depending on conversation length
- Simple Q&A (0 steps): ~$0.02

**89% of cost is input tokens.** Output is relatively cheap.

## Optimizations

### 1. Model Tiering Defaults

Default agents to a cheaper model instead of Opus:

| Task Complexity | Recommended Model | Input Cost |
|----------------|-------------------|-----------|
| Simple Q&A, echo, greetings | Haiku 4.5 | $1/MTok |
| General coding, writing, analysis | Sonnet 4 | $3/MTok |
| Complex reasoning, architecture | Opus 4.6 | $5/MTok |
| Routing, classification | GPT-4.1-nano | $0.10/MTok |

**Change default model** from Opus to Sonnet 4 in template/agent.json and GlobalConfig.

### 2. Tool Schema Optimization

Currently all 15+ tools are sent on every LLM call, even if the permission profile denies most of them. Only send tool schemas for tools the agent can actually use:

```python
# Before building tool schemas, filter by permission profile
allowed_tools = [t for t in tools.list_all() if check(format_action(t.name, ""), profile) != PermissionResult.DENY]
tool_schemas = tools_to_anthropic(allowed_tools)
```

A `locked` profile agent (view-only) would send 1 tool schema instead of 15 — saving ~2,400 tokens per call.

### 3. Compact Tool Descriptions

Review tool descriptions for verbosity. Each word in a description costs tokens on every call. Trim descriptions to the minimum needed for the LLM to use the tool correctly.

### 4. Lazy Doc Loading

Only include agent docs in the system prompt if the docs directory has content. Skip the `## Agent Documentation` header for agents with empty docs (saves ~50-100 tokens of header text).

### 5. Available Models List Cap

Currently sends up to 20 model names. Most agents don't need to know all available models. Reduce to 5 or remove entirely — agents can use `self_edit_model` to discover models when needed.

### 6. Cost Tracking and Alerts

Add per-agent cost tracking:
- Track cumulative input/output tokens per agent per day in SQLite
- Add `/agent cost <name>` command to show daily/weekly spend
- Optional: configurable cost cap per agent (stop processing when exceeded)

## Acceptance Criteria

- [ ] Default model changed to Sonnet 4 (in template and GlobalConfig)
- [ ] Tool schemas filtered by permission profile before sending
- [ ] Tool descriptions reviewed and trimmed
- [ ] Empty docs directories don't add doc header to system prompt
- [ ] Available models list capped at 5
- [ ] Per-agent cost tracking in SQLite
- [ ] `/agent cost` command implemented
- [ ] All existing tests pass

## Files

| File | Action |
|------|--------|
| `template/agent.json` | Modify — change default model |
| `src/chorus/config.py` | Modify — change GlobalConfig default model |
| `src/chorus/llm/tool_loop.py` | Modify — filter tools by permission profile |
| `src/chorus/agent/context.py` | Modify — lazy doc loading, model list cap |
| `src/chorus/tools/registry.py` | Modify — trim tool descriptions |
| `src/chorus/storage/db.py` | Modify — add cost tracking table |
| `src/chorus/commands/agent_commands.py` | Modify — add `/agent cost` command |

## Notes

- Model tiering is the single biggest cost lever. Moving from Opus ($5/MTok) to Sonnet ($3/MTok) saves 40% with minimal quality loss for most tasks.
- Tool schema filtering is the second biggest — removing tools the agent can't use saves tokens without changing behavior.
- These optimizations stack with prompt caching (TODO 023). Cached + filtered tool schemas = maximum savings.
