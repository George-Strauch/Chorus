# TODO 024 — Simplified Branch Routing (Drop Haiku Router)

> **Status:** PENDING
> **Priority:** High — reduces complexity, removes extra API call per message, saves cost
> **Dependencies:** 006 (execution threads), 008 (LLM integration)

## Objective

Remove the Haiku/gpt-4o-mini interjection router and replace it with a simple user-interaction-based routing model:

- **New message (non-reply):** Always creates a new branch.
- **Reply to a bot message:** Continues that branch — either injects into the tool loop if running, or resumes if completed.

This eliminates an entire LLM call per message when the agent is busy, removes router.py, and makes routing behavior predictable and user-controlled.

## Problem

The current router:
1. Adds latency (extra LLM call before the actual work starts)
2. Costs money (Haiku/gpt-4o-mini tokens for every message to a busy agent)
3. Is unpredictable (the LLM's INJECT vs NEW_THREAD decision may not match user intent)
4. Adds code complexity (`router.py`, `_route_interjection()`, `create_router_provider()`, `get_router_model()`)
5. Uses the router provider for branch summaries too — those should be moved elsewhere

## New Routing Model

```
User sends a message to agent channel:
├─ Is it a reply to a bot message?
│   ├─ YES → Find the branch that owns that bot message
│   │   ├─ Branch is RUNNING → Inject into its queue
│   │   └─ Branch is COMPLETED/IDLE → Resume the branch (start tool loop)
│   └─ NO (new message, not a reply)
│       └─ Always create a new branch, start tool loop
```

### What Changes

| Current Behavior | New Behavior |
|-----------------|-------------|
| Non-reply to idle agent → continues main thread | Non-reply → always new branch |
| Non-reply to busy agent → router decides INJECT or NEW_THREAD | Non-reply to busy agent → always new branch |
| Reply to bot message → continues that branch | Reply to bot message → continues that branch (unchanged) |
| Router LLM call on every message to busy agent | No router LLM call ever |
| "Main thread" concept exists | No main thread concept — every non-reply starts fresh |

### Benefits

- **Predictable:** Users control routing through their interaction pattern (reply = continue, new message = new topic)
- **Cheaper:** No router API calls
- **Simpler:** ~100 lines of code removed (router.py, routing logic in bot.py)
- **Faster:** No latency from router LLM call

### Trade-offs

- Users must explicitly reply to continue a conversation. New messages always start fresh branches.
- If a user forgets to reply and sends a new message, it creates a new branch instead of continuing. This is acceptable — the user can reply to the previous bot message to get back to that branch.
- Branch summaries currently use the router provider — need a new approach (see below).

## Branch Summaries

The router provider is also used for generating 5-word branch summaries. Options:

1. **Use the agent's own model** — slightly more expensive but no separate provider needed
2. **Generate summary from the user's first message** — just take the first N words, no LLM call needed
3. **Keep a cheap provider just for summaries** — defeats the purpose of removing the router

**Recommendation:** Option 2 — extract the first ~5 words from the user's initial message as the branch label. Simple, free, and descriptive enough for `/thread list`. If the user sends "Fix the authentication bug in login.py", the summary becomes "Fix the authentication bug in...".

## Implementation

### 1. Delete `src/chorus/llm/router.py`

Remove the entire file.

### 2. Simplify `on_message` in `bot.py`

Replace the main-thread routing block (lines ~311-388) with:

```python
# ── Routing ─────────────────────────────────────────────────
# Reply → continue that branch
# New message → always new branch

if message.reference and message.reference.message_id:
    # Reply-based routing (existing, unchanged)
    thread = tm.route_message(message.reference.message_id)
    if thread is not None:
        # ... inject or resume (existing logic)
        return

# Non-reply: always create a new branch
thread = await tm.create_branch(
    {"role": "user", "content": message.content}
)
# ... persist, start tool loop
```

### 3. Remove Router References from `bot.py`

- Remove `_router_provider`, `_router_model` attributes
- Remove `_route_interjection()` method
- Remove `create_router_provider` import
- Remove router initialization in `__init__`

### 4. Update Branch Summary Generation

Replace `_generate_branch_summary()` with a simple text extraction:

```python
async def _generate_branch_summary(self, agent_name: str, branch_id: int, first_message: str) -> None:
    summary = first_message[:50].strip()
    if len(first_message) > 50:
        summary += "..."
    await self.db.set_branch_summary(agent_name, branch_id, summary)
```

### 5. Remove `/break-context` Command

With no main thread concept, `/break-context` (which detaches the main thread) is no longer meaningful. Remove it.

### 6. Remove Idle Timeout Logic

The idle timeout that forces a new branch after 3 hours is no longer needed — every new message already creates a new branch.

## Acceptance Criteria

- [ ] `src/chorus/llm/router.py` deleted
- [ ] Non-reply messages always create a new branch
- [ ] Reply messages continue the referenced branch (inject if running, resume if not)
- [ ] No router LLM calls anywhere in the codebase
- [ ] Branch summaries generated from the user's first message (no LLM call)
- [ ] `/break-context` command removed
- [ ] Idle timeout logic removed from `on_message`
- [ ] All imports/references to router.py cleaned up
- [ ] All existing routing tests updated
- [ ] New tests: non-reply always creates branch, reply continues branch, reply to running branch injects

## Files

| File | Action |
|------|--------|
| `src/chorus/llm/router.py` | Delete |
| `src/chorus/bot.py` | Modify — simplify on_message routing, remove router refs |
| `src/chorus/commands/thread_commands.py` | Modify — remove `/break-context` |
| `tests/test_llm/test_router.py` | Delete (if exists) or repurpose |
| `tests/test_bot.py` | Update routing tests |

## Notes

- This is a behavioral change for existing users — document it in a changelog
- The `inject_queue` mechanism on `ExecutionThread` is preserved (reply to a running branch still injects)
- `ThreadManager.get_main_thread()` can be removed or kept for backward compat — it just won't be called from bot.py
