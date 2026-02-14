# TODO 022: Tool Call Chain Optimization

> **Status:** PENDING
> **Priority:** High — reduces latency and API cost for multi-tool workflows
> **Dependencies:** 008 (LLM integration), tool_loop.py

## Objective

Optimize the tool loop to execute chains of tool calls without unnecessary LLM round-trips. Currently, every tool call requires a full LLM inference cycle: the model returns tool calls, we execute them, send results back, and the model decides what to do next. Many tool sequences are predictable and don't need the model's judgment between steps.

## Problem

A typical agent workflow looks like:

```
User: "Read main.py and tests/test_main.py and fix the bug"

LLM call 1 → tool_use: view(path="main.py")        # 200K input tokens
Tool result → file contents
LLM call 2 → tool_use: view(path="tests/test_main.py")  # 200K+ input tokens
Tool result → file contents
LLM call 3 → tool_use: str_replace(...)             # 200K+ input tokens
Tool result → success
LLM call 4 → tool_use: bash(command="pytest")       # 200K+ input tokens
Tool result → test output
LLM call 5 → final text response                    # 200K+ input tokens
```

That's 5 LLM calls at ~200K input tokens each = ~1M input tokens. But calls 1 and 2 are independent reads — they could have been batched. The model already requested both; it just had to wait for the API to process them sequentially.

**The real cost:** Anthropic charges $15/MTok input for Claude Opus. A 5-call chain at 200K each costs ~$15. If we can reduce it to 3 calls, that's ~$9 — a 40% savings.

## Approach: Parallel Tool Execution

### What the API Already Gives Us

Both Anthropic and OpenAI APIs support **multiple tool calls in a single response**. The model can return:

```json
{
  "tool_calls": [
    {"id": "tc_1", "name": "view", "arguments": {"path": "main.py"}},
    {"id": "tc_2", "name": "view", "arguments": {"path": "tests/test_main.py"}}
  ]
}
```

Our current `run_tool_loop` already handles this — it executes all tool calls in the response. But it does so **sequentially**. The first optimization is straightforward:

### Phase 1: Parallel Execution of Independent Tool Calls

When the model returns multiple tool calls in one response, execute them concurrently with `asyncio.gather()` instead of sequentially.

**Current code** (tool_loop.py ~line 460):
```python
for tc in response.tool_calls:
    result = await _handle_tool_call(tc, tools, ctx, ask_callback)
    working_messages.append({"role": "tool", ...})
```

**Proposed:**
```python
# Execute all tool calls concurrently
tasks = [_handle_tool_call(tc, tools, ctx, ask_callback) for tc in response.tool_calls]
results = await asyncio.gather(*tasks)

# Append results in order
for tc, result in zip(response.tool_calls, results):
    working_messages.append({"role": "tool", ...})
```

**Constraints:**
- File locks already protect concurrent access (per-agent per-file `asyncio.Lock`)
- Bash commands may have ordering dependencies — but if the model sent them in one batch, it's asserting they're independent
- Permission ASK callbacks must still be sequential (can't show 5 Discord permission prompts at once)

**Implementation:**
- If any tool call requires ASK permission, fall back to sequential execution for the entire batch
- Otherwise, `asyncio.gather()` all of them

### Phase 2: Speculative Pre-execution (Future)

More aggressive: analyze tool call patterns and speculatively execute likely follow-ups without waiting for the LLM.

**Pattern detection examples:**

| Pattern | Pre-execution |
|---------|---------------|
| `view(path=X)` → LLM often follows with `str_replace(path=X, ...)` | No — edit depends on LLM seeing the file |
| `bash("pytest")` → if tests fail, LLM reads the failing file | Could pre-read files mentioned in traceback |
| `view(A)` then `view(B)` then `view(C)` | If model requests view(A), and recent context shows it asked about B and C, pre-fetch them |
| `git_diff` → `git_commit` | No — commit message depends on LLM reading the diff |

This is complex and error-prone. **Defer to a future iteration.** The parallel execution in Phase 1 is the high-value, low-risk win.

### Phase 3: Reduced Round-Trips via Tool Call Hinting (Future)

Encourage the model to batch tool calls by including system prompt guidance:

```
When you need to read multiple files, request all views in a single response
rather than one at a time. You can make multiple tool calls simultaneously.
```

This is a prompt engineering change, not a code change. Can be added to the default system prompt or agent template.

## Acceptance Criteria

### Phase 1 (this TODO)
- [ ] Multiple tool calls in one LLM response execute concurrently via `asyncio.gather()`
- [ ] Sequential fallback when any tool call requires ASK permission
- [ ] Tool results appended in the same order as tool calls (API requirement)
- [ ] Events still fire for each tool call (TOOL_CALL_START/COMPLETE)
- [ ] File locking prevents race conditions on concurrent file ops
- [ ] All existing tests pass
- [ ] New tests cover: parallel execution, ASK fallback, ordering, error in one doesn't kill others

## Implementation Notes

### File: `src/chorus/llm/tool_loop.py`

The main change is in `run_tool_loop()`, in the tool execution section (~line 455-480):

```python
# Check if any tool call needs ASK permission
needs_sequential = False
for tc in response.tool_calls:
    tool = tools.get(tc.name)
    if tool is None:
        continue
    action = _build_action_string(tc.name, tc.arguments)
    perm = check(action, ctx.profile)
    if perm is PermissionResult.ASK:
        needs_sequential = True
        break

if needs_sequential or len(response.tool_calls) == 1:
    # Sequential execution (current behavior)
    for tc in response.tool_calls:
        ...
else:
    # Parallel execution
    async def _run_one(tc: ToolCall) -> tuple[ToolCall, str]:
        await _fire_event(on_event, ToolLoopEvent(
            type=ToolLoopEventType.TOOL_CALL_START, ...
        ))
        result = await _handle_tool_call(tc, tools, ctx, ask_callback)
        await _fire_event(on_event, ToolLoopEvent(
            type=ToolLoopEventType.TOOL_CALL_COMPLETE, ...
        ))
        return tc, result

    pairs = await asyncio.gather(
        *[_run_one(tc) for tc in response.tool_calls]
    )
    for tc, result in pairs:
        working_messages.append({"role": "tool", ...})
```

### Error Handling

If one tool call in a parallel batch fails (raises exception), `asyncio.gather()` with `return_exceptions=True` prevents the others from being cancelled. Each result is either a string or an exception, handled individually.

### Testing

- Test parallel execution reduces wall-clock time (mock tools with `asyncio.sleep`)
- Test ASK permission triggers sequential fallback
- Test ordering preserved in tool results
- Test one failure doesn't prevent others from completing
- Test events fire correctly for parallel calls

## Files

| File | Action |
|------|--------|
| `src/chorus/llm/tool_loop.py` | Modify — parallel tool execution |
| `tests/test_llm/test_tool_loop.py` | Modify — add parallel execution tests |
| `template/agent.json` | Consider — add batching hint to default system prompt |

## Risks

- **Race conditions in bash**: Two concurrent `bash` calls in the same workspace could interfere. File locking only protects file ops, not arbitrary shell commands. Mitigation: the model chose to batch them, implying independence.
- **Discord rate limits on ASK**: If we need to show multiple ASK prompts, doing them sequentially is correct. The fallback handles this.
- **Error propagation**: A failed tool in a batch shouldn't prevent other results from being sent to the LLM. Use `return_exceptions=True`.
