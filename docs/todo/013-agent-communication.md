# TODO 013 — Inter-Agent Communication

> **Status:** DONE

## Objective

Enable agents to collaborate by sending messages to each other, reading each other's docs, and discovering available agents. Communication is fire-and-forget: the sending agent continues working while the target processes the message as a new branch.

## Acceptance Criteria

- [x] `send_to_agent(target_agent, message)` delivers an attributed message to the target agent, spawning a new branch in the target's channel.
- [x] Target agent runs with its own permissions (not admin).
- [x] Self-send is rejected with a clear error.
- [x] Missing target agent is rejected with a clear error.
- [x] `read_agent_docs(target_agent)` reads all `.md` files from the target's `docs/` directory.
- [x] `list_agents()` returns all agents (excluding self) with model and description from `docs/README.md`.
- [x] Permission category `agent_comm` with action strings: `send <target>`, `read_docs <target>`, `list`.
- [x] Standard preset ALLOWs all `agent_comm` actions (no confirmation needed).
- [x] Tools registered in the default registry with proper JSON schemas.
- [x] `bot` injectable added to `ToolExecutionContext` and `_execute_tool`.
- [x] `spawn_agent_branch` method on `ChorusBot` for delivering inter-agent messages.
- [x] 30 tests passing across 6 test classes.

## What Was Implemented

### New file: `src/chorus/agent/communication.py`

Three handler functions + helper:

- **`send_to_agent(target_agent, message, *, agent_name, bot, chorus_home)`** — validates target exists & not self, formats attributed message, calls `bot.spawn_agent_branch()`, returns JSON result.
- **`read_agent_docs(target_agent, *, agent_name, chorus_home)`** — reads all `.md` files from target's `docs/`, returns content dict.
- **`list_agents(*, agent_name, chorus_home, db)`** — lists agent directories excluding self, extracts description + model.
- **`_extract_first_paragraph(markdown)`** — skips headings/emphasis/blockquotes, returns first prose line (capped at 200 chars).

### Modified files

- **`tool_loop.py`** — Added `bot` to `_CONTEXT_INJECTED_PARAMS`, `ToolExecutionContext`, `_execute_tool` injection; added `agent_comm` category mapping and action string builder.
- **`bot.py`** — Added `spawn_agent_branch()` method (like `spawn_hook_branch` but `is_admin=False`); wired `bot=self` into `ToolExecutionContext`.
- **`engine.py`** — Added `r"tool:agent_comm:.*"` to standard preset's allow list.
- **`registry.py`** — Registered `send_to_agent`, `read_agent_docs`, `list_agents` with JSON schemas.

### New file: `tests/test_agent/test_communication.py`

6 test classes, 30 tests:
- `TestSendToAgent` (6) — delivery, self-rejection, missing target, no bot, spawn called, spawn failure
- `TestReadAgentDocs` (5) — content returned, self-rejection, missing agent, multiple files, empty docs
- `TestListAgents` (4) — lists agents, excludes self, includes descriptions/model, empty dir
- `TestExtractFirstParagraph` (7) — heading skip, italic skip, empty content, truncation, plain, bold skip, blockquote skip
- `TestPermissionIntegration` (5) — tools registered, allowed in standard/open, category mapping, action strings
- `TestToolExecution` (3) — end-to-end via `_execute_tool` for send, list, read_docs

## What Was NOT Implemented (deferred)

- **`request_from_agent`** (blocking request/response) — deferred; fire-and-forget is sufficient for now.
- **Circular communication detection** — deferred; not needed for fire-and-forget.
- **Rate limiting** — deferred; can be added if message storms become a real problem.

## Dependencies

- **001-010**: All core functionality (complete).
