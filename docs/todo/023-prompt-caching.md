# TODO 023 — Anthropic Prompt Caching

> **Status:** PENDING
> **Priority:** High — 60-90% reduction in input token costs
> **Dependencies:** 008 (LLM integration)

## Objective

Add `cache_control` headers to Anthropic API calls so that static content (system prompt, agent docs, tool definitions) is cached across tool loop iterations. Currently every LLM call pays full input price for the entire context — with caching, repeated content costs 90% less.

## Problem

Billing shows **zero cache tokens** because we never set `cache_control` on any content blocks. A typical tool loop iteration sends ~4,500 tokens of static content (system prompt + docs + tool schemas). Over a 5-step loop on Opus 4.6:

- **Without caching:** 4,500 × 5 × $5/MTok = **$0.1125** (static portion only)
- **With caching:** 4,500 × $6.25/MTok (write) + 4,500 × 4 × $0.50/MTok (read) = **$0.0371**
- **Savings: 67%** on the static portion

For larger contexts with more history that stays stable across iterations, savings approach 90%.

## Anthropic Cache Pricing

| Model | Base Input | Cache Write (1.25x) | Cache Read (0.1x) | Savings on Read |
|-------|-----------|--------------------|--------------------|-----------------|
| Opus 4.6 | $5/MTok | $6.25/MTok | $0.50/MTok | 90% |
| Sonnet 4 | $3/MTok | $3.75/MTok | $0.30/MTok | 90% |
| Haiku 4.5 | $1/MTok | $1.25/MTok | $0.10/MTok | 90% |

Cache TTL: 5 minutes (default, `{"type": "ephemeral"}`), refreshed each time it's used. Tool loop calls happen within seconds of each other, so the cache will always be warm within a single turn.

### Minimum Cacheable Sizes

- Haiku 4.5, Opus 4.5/4.6: **4,096 tokens**
- Sonnet 4/4.5, Opus 4/4.1: **1,024 tokens**

The system prompt + docs + tool schemas should exceed 4,096 tokens in most cases. If not, the API silently ignores the `cache_control` marker (no error).

## Implementation

### 1. Convert System Prompt to Block Format

In `AnthropicProvider.chat()`, change from:

```python
system=system_text
```

To:

```python
system=[
    {"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}},
]
```

### 2. Cache Tool Definitions

Add `cache_control` to the last tool in the tools list (Anthropic caches the prefix up to each breakpoint):

```python
if tools:
    tools[-1]["cache_control"] = {"type": "ephemeral"}
```

### 3. Track Cache Metrics in Usage

Extend the `Usage` dataclass:

```python
@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
```

Parse these from Anthropic's response `usage` field.

### 4. Display Cache Metrics

Update `format_response_footer()` in `ui/status.py` to show cache hits when available:

```
branch #5 · 3 steps · 1,234 in (890 cached) / 567 out · 12.5s
```

## Acceptance Criteria

- [ ] Anthropic API calls include `cache_control` on system prompt content block
- [ ] Anthropic API calls include `cache_control` on the last tool definition
- [ ] `Usage` dataclass tracks `cache_creation_input_tokens` and `cache_read_input_tokens`
- [ ] Status footer displays cache metrics when non-zero
- [ ] OpenAI provider is unaffected (OpenAI caching is automatic, no changes needed)
- [ ] All existing tests pass
- [ ] New tests verify cache_control blocks are present in Anthropic API calls
- [ ] New test verifies Usage parsing includes cache fields

## Files

| File | Action |
|------|--------|
| `src/chorus/llm/providers.py` | Modify — add cache_control to system + tools, parse cache usage |
| `src/chorus/llm/tool_loop.py` | Minor — ensure Usage addition handles new fields |
| `src/chorus/ui/status.py` | Modify — display cache metrics in footer |
| `tests/test_llm/test_providers.py` | Add — verify cache_control in API calls |
| `tests/test_llm/test_tool_loop.py` | Add — verify cache usage aggregation |

## Notes

- Up to 4 `cache_control` breakpoints per request. We use 2 (system prompt, tools). A third could be added to the last conversation message for multi-turn caching.
- This only affects Anthropic. OpenAI does automatic caching at 50% discount — no code changes needed.
- The cache is per-model, per-organization. Switching models mid-conversation creates a new cache entry (acceptable).
