# TODO 020 — Fix Web Search Server-Side Tool Handling

> **Status:** PENDING
> **Priority:** High — feature is broken in production
> **Dependencies:** 008 (LLM integration)

## Background

Web search was implemented in commit `a2b9ccf` using Anthropic's **server-side tool** (`web_search_20250305`) instead of a third-party API (Tavily) as originally planned. The implementation successfully:

- Added `web_search` boolean to the Agent model
- Created `self_edit_web_search` tool for agents to toggle the feature
- Injected the server-side tool spec into the tools list when enabled
- Added `_raw_content` preservation in providers.py for round-tripping

**However, it is broken.** When Anthropic processes a web search and returns results, the tool loop tries to look up `web_search` in the local tool registry, fails, and returns `{"error": "Unknown tool: web_search"}`.

## The Bug

The tool loop in `tool_loop.py` treats all tool calls the same way:

```python
tool = tools.get(tc.name)  # Looks for "web_search" in registry
if tool is None:
    return json.dumps({"error": f"Unknown tool: {tc.name}"})
```

But `web_search` is a **server-side tool** — Anthropic's API executes it and returns the results inline. There's no local handler to call. The tool loop doesn't distinguish between "tool calls I need to execute locally" and "tool calls the API already executed server-side."

## Fix

Special-case server-side tool results in `_execute_tool` (or its caller in the tool loop). When the tool name matches a known server-side tool (`web_search`), skip local execution and pass through the raw result from the API response.

### Approach

1. In the tool loop, after receiving an LLM response, check if any tool calls correspond to server-side tools
2. For server-side tools, extract the result from `response._raw_content` (already preserved) instead of calling the local registry
3. The tool result message should contain the server-side result as-is

### Key Files

| File | Action |
|------|--------|
| `src/chorus/llm/tool_loop.py` | Modify — skip local execution for server-side tools |
| `src/chorus/llm/providers.py` | Review — ensure `_raw_content` captures `web_search_tool_result` blocks |
| `tests/test_llm/test_tool_loop.py` | Add — test that simulates Anthropic returning a `web_search` tool call + result |

## Acceptance Criteria

- [ ] When `web_search` is enabled and Anthropic returns search results, they are passed through to the conversation correctly
- [ ] The tool loop does not try to execute server-side tools locally
- [ ] Permission checks still apply (the existing pre-check before injection is sufficient)
- [ ] Existing tests continue to pass
- [ ] New test: mock Anthropic returning `server_tool_use` + `web_search_tool_result` content blocks, verify they round-trip through the tool loop
- [ ] New test: verify unknown server-side tools are handled gracefully

## Notes

- The `_raw_content` field on `LLMResponse` already captures server-side tool blocks — this was added in the original web search commit
- Consider future server-side tools (Anthropic may add more) — the fix should be generic enough to handle any server-side tool, not just `web_search`
- TODO 020 originally planned a third-party API approach (Tavily). That approach remains viable as a future enhancement but is not needed for the fix
