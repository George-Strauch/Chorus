# TODO 013 — Agent Communication (Future Phase)

> **Status:** PENDING

## Objective

Enable inter-agent messaging: agents can send messages to other agent channels, request work from other agents, and receive responses. This transforms Chorus from a collection of independent agents into a collaborative multi-agent system where agents can delegate tasks, share information, and coordinate workflows.

## Acceptance Criteria

- An agent can send a message to another agent's channel using a `send_to_agent(target_agent, message)` tool.
- The target agent receives the message as a regular channel message and can respond to it.
- Responses are routed back to the originating agent's context.
- Agents can discover other agents via a `list_agents()` tool that returns names and descriptions (from each agent's `docs/README.md`).
- Permission check applies: `tool:agent_comm:send <target_agent>`. The standard profile asks for confirmation; open allows automatically.
- Communication is asynchronous — the sending agent continues working while waiting for a response.
- A `request_from_agent(target_agent, task_description, timeout?)` tool sends a message and waits for a response, with a configurable timeout.
- Communication history is logged in both agents' contexts and in the SQLite audit table.
- Circular communication (agent A asks agent B, which asks agent A) is detected and broken after a configurable depth (default 3).
- Rate limiting prevents message storms between agents (default 10 inter-agent messages per minute per agent).

## Tests to Write First

File: `tests/test_agent/test_communication.py` (future)
```
# Sending
test_send_to_agent_delivers_message
test_send_to_agent_permission_check
test_send_to_agent_nonexistent_target_fails
test_send_to_agent_message_appears_in_target_channel
test_send_to_agent_logged_in_audit

# Request/response
test_request_from_agent_waits_for_response
test_request_from_agent_timeout
test_request_from_agent_response_routed_back

# Discovery
test_list_agents_returns_all_agents
test_list_agents_includes_descriptions
test_list_agents_excludes_self

# Safety
test_circular_communication_detected
test_circular_communication_broken_at_max_depth
test_rate_limiting_between_agents
test_rate_limit_resets_after_window

# Context
test_communication_logged_in_sender_context
test_communication_logged_in_receiver_context
```

## Implementation Notes

1. **Module location:** `src/chorus/agent/communication.py` (new module) and new tools registered in the tool registry.

2. **send_to_agent implementation:**
   ```python
   async def send_to_agent(
       source_agent: str,
       target_agent: str,
       message: str,
       bot: ChorusBot,
       profile: PermissionProfile,
   ) -> CommResult:
       action = format_action("agent_comm", f"send {target_agent}")
       perm = check(action, profile)
       # ... permission handling ...
       target_channel = bot.get_agent_channel(target_agent)
       await target_channel.send(f"**From {source_agent}:** {message}")
       return CommResult(delivered=True, ...)
   ```

3. **Request/response pattern:** Use `asyncio.Event` or `asyncio.Queue` for response routing:
   - Sending agent posts a message with a unique request ID.
   - Target agent's response handler watches for messages that reference the request ID.
   - On response, the event is set and the sending agent's `request_from_agent` call returns.
   - On timeout, return a timeout error to the LLM so it can decide what to do.

4. **Circular communication detection:** Maintain a call stack in each message. When agent A sends to agent B, the message includes `call_chain: ["A"]`. When B sends to A, it becomes `call_chain: ["A", "B"]`. If the chain length exceeds `max_depth`, reject the send with an error explaining the circular reference.

5. **Agent discovery tool:** `list_agents()` reads all agent directories, extracts the first paragraph of each `docs/README.md`, and returns a list of `{name, description, model, status}`.

6. **Use cases this enables:**
   - A "project manager" agent delegates coding tasks to a "developer" agent.
   - A "reviewer" agent requests changes from the author agent.
   - A "research" agent gathers information and sends summaries to a "writing" agent.
   - An "ops" agent detects an issue and asks a "debug" agent to investigate.

7. **Message format in target channel:** Inter-agent messages should be visually distinct from user messages. Use an embed with a different color and a clear "From: <agent>" header.

8. **This is a future phase.** Do not implement until TODOs 001-010 are complete and stable. The design here should be validated against real multi-agent workflows before building.

## Dependencies

- **001-010**: All core functionality must be complete and stable.
