# TODO 018 — Planning Mode

> **Status:** PENDING

## Objective

Implement a planning mode that separates agent execution into two distinct phases: **context gathering** (read-only exploration) and **execution** (tool use with side effects). This mirrors Claude Code's plan mode — the agent first investigates, forms a plan, and presents it for approval before taking action. Planning mode can be entered explicitly via a slash command (`/plan`) or requested by the agent itself when it recognizes a task would benefit from upfront analysis.

## Motivation

Currently, agents jump straight into tool execution when they receive a message. For complex, multi-step, or ambiguous tasks this is risky: the agent may take irreversible actions before fully understanding the problem. Planning mode enforces a disciplined workflow:

1. **Gather context** — read files, search, explore, ask clarifying questions. No writes, no bash side effects, no git mutations.
2. **Present a plan** — structured summary of what the agent intends to do, why, and in what order.
3. **User approves/rejects/edits** — the plan is shown in Discord; user reacts or replies.
4. **Execute** — agent carries out the approved plan with full tool access.

This produces better outcomes for non-trivial tasks, gives the user visibility into the agent's reasoning, and reduces wasted work from misunderstood requirements.

## Acceptance Criteria

### Entering Planning Mode

- `/plan` slash command enters planning mode for the channel's agent. Accepts an optional `task` argument describing what to plan for.
- `/plan` while already in planning mode is a no-op (responds with current plan status).
- The agent can request planning mode from within the tool loop by calling a `enter_plan_mode` tool. This pauses execution, switches to planning phase, and notifies the user.
- When planning mode is entered, the agent's tool access is restricted to **read-only tools** (view, glob/search, bash with read-only commands) plus a set of **planning-specific tools**.

### Context Gathering Phase

- The agent operates in a restricted tool environment:
  - **Allowed:** `view` (file read), `bash_execute` (read-only subset — no writes, no mutations), search/glob operations, `ask_user` (ask clarifying questions in Discord).
  - **Blocked:** `create_file`, `str_replace`, `bash_execute` (write commands), `git_*` (all git tools), `self_edit_*` (all self-edit tools).
- Read-only bash enforcement: a blocklist of write/mutating commands (similar to the existing bash blocklist) is applied during planning mode. Commands like `cat`, `ls`, `find`, `grep`, `head`, `tail`, `tree`, `wc`, `diff`, `file`, `stat` are allowed. Commands like `rm`, `mv`, `cp`, `mkdir`, `touch`, `chmod`, `tee`, write redirections (`>`, `>>`), `pip install`, `git push/commit/checkout` etc. are blocked.
- The agent can call `ask_user(question)` to ask the user a clarifying question via Discord. The user's reply is fed back into the planning context.
- No iteration limit on the context gathering phase (bounded by the normal tool loop max_iterations). The agent decides when it has enough context.

### Plan Creation

- When the agent has gathered enough context, it calls `submit_plan(plan)` — a planning-specific tool.
- The plan is a structured object:
  ```python
  @dataclass
  class Plan:
      summary: str              # 1-3 sentence overview
      steps: list[PlanStep]     # Ordered steps
      questions: list[str]      # Unresolved questions for the user (if any)
      context_notes: str        # Key findings from the gathering phase

  @dataclass
  class PlanStep:
      description: str          # What this step does
      tools: list[str]          # Which tools it will use
      risk: str                 # "low", "medium", "high" — signals to the user
  ```
- The plan is rendered as a Discord embed with: summary, numbered steps (with risk indicators), unresolved questions, and approve/reject buttons.
- The plan is persisted to the agent's context (SQLite) so it survives restarts.

### Plan Approval

- A `PlanApprovalView` (discord.ui.View) is sent with the plan embed:
  - **Approve** button — proceed to execution phase with the plan as guidance.
  - **Reject** button — cancel planning mode, return to normal operation.
  - **Edit** button — user can reply with modifications; agent revises the plan and re-submits.
- Approval/rejection is logged to the audit table.
- If the user doesn't respond within a configurable timeout (default: 30 minutes), planning mode is cancelled automatically.

### Execution Phase

- On approval, the agent enters execution with:
  - Full tool access restored (subject to normal permission profile).
  - The approved plan injected into the system prompt as structured guidance.
  - Each plan step tracked — the agent should reference which step it's executing.
- The agent is expected to follow the plan but is not rigidly locked to it. If it encounters something unexpected, it can adapt, but significant deviations should be communicated to the user.
- When all plan steps are completed (or the agent determines the task is done), planning mode ends automatically and the agent returns to normal operation.

### Agent-Initiated Planning

- A tool `enter_plan_mode(reason)` is available in the normal tool registry. When the agent calls it:
  1. The current tool loop iteration pauses.
  2. A Discord message notifies the user: "I'd like to plan this out before proceeding. Reason: {reason}"
  3. The agent's tool access switches to read-only.
  4. The agent continues in the same thread but now in planning mode.
- The agent's system prompt should encourage entering planning mode for complex or ambiguous tasks. This is a guideline, not enforcement — the agent can always choose to act directly.

### Slash Command

```
/plan [task]                    # Enter planning mode (optional task description)
/plan status                    # Show current plan status (gathering/submitted/executing)
/plan cancel                    # Cancel planning mode, return to normal
```

## Tests to Write First

File: `tests/test_agent/test_planning.py`
```
# Planning mode lifecycle
test_enter_planning_mode_restricts_tools
test_exit_planning_mode_restores_tools
test_plan_cancel_restores_normal_mode
test_plan_already_active_is_noop

# Context gathering phase
test_view_tool_allowed_in_planning
test_bash_read_only_allowed_in_planning
test_bash_write_blocked_in_planning
test_create_file_blocked_in_planning
test_str_replace_blocked_in_planning
test_git_tools_blocked_in_planning
test_self_edit_blocked_in_planning
test_ask_user_allowed_in_planning

# Plan submission
test_submit_plan_creates_plan_object
test_submit_plan_validates_structure
test_submit_plan_persisted_to_db
test_submit_plan_sends_discord_embed

# Plan approval
test_plan_approve_enters_execution
test_plan_reject_cancels_mode
test_plan_edit_allows_revision
test_plan_timeout_cancels_mode
test_plan_approval_logged_to_audit

# Execution phase
test_execution_has_full_tool_access
test_plan_injected_into_system_prompt
test_execution_completion_exits_planning_mode

# Agent-initiated planning
test_enter_plan_mode_tool_switches_to_planning
test_enter_plan_mode_notifies_user
test_enter_plan_mode_reason_displayed

# Slash commands
test_plan_command_enters_mode
test_plan_command_with_task
test_plan_status_shows_current_state
test_plan_cancel_exits_mode
```

File: `tests/test_tools/test_planning_tools.py`
```
# Read-only bash enforcement
test_planning_bash_allows_cat
test_planning_bash_allows_ls
test_planning_bash_allows_grep
test_planning_bash_allows_find
test_planning_bash_blocks_rm
test_planning_bash_blocks_write_redirect
test_planning_bash_blocks_pip_install
test_planning_bash_blocks_git_push

# Planning tools
test_submit_plan_tool_registered
test_enter_plan_mode_tool_registered
test_ask_user_tool_registered
test_submit_plan_tool_schema
```

## Implementation Notes

1. **Module location:** `src/chorus/agent/planning.py` — core planning state machine.

2. **Planning state enum:**
   ```python
   class PlanningPhase(Enum):
       INACTIVE = "inactive"         # Normal operation
       GATHERING = "gathering"       # Read-only context gathering
       PLAN_SUBMITTED = "submitted"  # Waiting for user approval
       EXECUTING = "executing"       # Approved plan being carried out
   ```

3. **PlanningManager class:**
   ```python
   class PlanningManager:
       def __init__(self, agent_name: str, db: Database):
           self.agent_name = agent_name
           self.phase: PlanningPhase = PlanningPhase.INACTIVE
           self.plan: Plan | None = None
           self.task_description: str | None = None
           self.gathering_start: datetime | None = None

       async def enter(self, task: str | None = None) -> None: ...
       async def submit_plan(self, plan: Plan) -> None: ...
       async def approve(self) -> None: ...
       async def reject(self) -> None: ...
       async def cancel(self) -> None: ...
       def is_active(self) -> bool: ...
       def get_tool_filter(self) -> Callable[[str], bool]: ...
   ```

4. **Tool filtering integration:** The `PlanningManager.get_tool_filter()` returns a predicate that the tool registry uses to filter available tools before each LLM call. During `GATHERING` phase, only read-only tools pass. During `EXECUTING` and `INACTIVE`, all tools pass (subject to normal permissions).

   Hook into `run_tool_loop()`:
   ```python
   # Before building tool schemas for the LLM call
   available_tools = registry.get_tools()
   if planning_manager and planning_manager.is_active():
       tool_filter = planning_manager.get_tool_filter()
       available_tools = [t for t in available_tools if tool_filter(t.name)]
   ```

5. **Read-only bash enforcement:** During planning mode, `bash_execute` is still available but with an additional blocklist layer. Create a `PLANNING_BASH_BLOCKLIST` that extends the existing blocklist with write/mutation commands. Apply it as a pre-execution check in the tool handler (or as a wrapper).

   ```python
   PLANNING_BASH_BLOCKLIST = [
       # File mutations
       "rm", "mv", "cp", "mkdir", "rmdir", "touch", "chmod", "chown",
       "ln", "install", "mktemp",
       # Write redirections detected by parser
       # Package management
       "pip", "npm", "yarn", "cargo", "apt", "pacman",
       # Git write operations
       "git push", "git commit", "git checkout", "git reset",
       "git merge", "git rebase", "git stash",
       # Editors
       "nano", "vim", "vi", "sed -i", "tee",
   ]
   ```

6. **Plan rendering:** Use `discord.Embed` for the plan display:
   - Title: "Plan: {summary}"
   - Fields: one per step, with risk emoji (green/yellow/red circle)
   - Footer: "React to approve or reject"
   - Attached `PlanApprovalView` with Approve/Reject/Edit buttons

7. **Plan injection during execution:** When `phase == EXECUTING`, `build_llm_context()` includes the plan as an additional system prompt section:
   ```
   <approved_plan>
   You are executing an approved plan. Follow these steps:

   1. [Step description] (tools: create_file, str_replace)
   2. [Step description] (tools: bash_execute)
   ...

   You may adapt if you encounter unexpected issues, but communicate
   significant deviations to the user.
   </approved_plan>
   ```

8. **Persistence:** Store planning state in SQLite:
   ```sql
   CREATE TABLE planning_state (
       agent_name TEXT PRIMARY KEY,
       phase TEXT NOT NULL DEFAULT 'inactive',
       task_description TEXT,
       plan_json TEXT,              -- JSON-serialized Plan
       gathering_start TEXT,
       submitted_at TEXT,
       approved_at TEXT,
       approved_by INTEGER          -- Discord user ID
   );
   ```
   On bot restart, restore planning state from this table. If phase was `GATHERING` or `PLAN_SUBMITTED`, resume from that point.

9. **Slash command registration:** New cog `commands/plan_commands.py`:
   ```python
   class PlanCog(commands.GroupCog, group_name="plan"):
       @app_commands.command(name="start")
       async def plan_start(self, interaction, task: str | None = None): ...

       @app_commands.command(name="status")
       async def plan_status(self, interaction): ...

       @app_commands.command(name="cancel")
       async def plan_cancel(self, interaction): ...
   ```

10. **Agent-initiated entry:** The `enter_plan_mode` tool handler:
    ```python
    async def enter_plan_mode(
        reason: str,
        *,
        agent_name: str,
        db: Database,
    ) -> str:
        planning_manager = get_planning_manager(agent_name)
        if planning_manager.is_active():
            return "Already in planning mode."
        await planning_manager.enter(task=reason)
        return f"Entered planning mode. Reason: {reason}. Tools are now restricted to read-only. Gather context, then call submit_plan() when ready."
    ```

11. **System prompt guidance:** Add to the default agent system prompt:
    ```
    For complex or ambiguous tasks, consider entering planning mode first
    by calling enter_plan_mode(reason). This restricts your tools to
    read-only while you investigate, then lets you present a structured
    plan for user approval before taking action. Use this for tasks that
    involve multiple files, architectural decisions, or where mistakes
    would be costly to undo.
    ```

12. **Thread interaction:** Planning mode is per-agent (not per-thread). When planning mode is active, all threads for that agent respect the tool restrictions. The main thread is typically the one doing the planning. If a new message arrives during planning, the interjection router handles it normally — the injected message becomes part of the planning context.

## Edge Cases

- **Planning mode + permission ASK overlap:** During gathering, if a read-only bash command still triggers an ASK (e.g., user has very restrictive permissions), the ASK flow works normally. Planning mode restricts the tool set, not the permission checks within allowed tools.
- **Bot restart during planning:** Restore from `planning_state` table. If `GATHERING`, resume with read-only tools. If `PLAN_SUBMITTED`, re-send the approval embed.
- **Multiple agents in planning mode:** Each agent has its own `PlanningManager`. No cross-agent interaction.
- **Planning mode timeout:** If gathering phase exceeds a configurable limit (default: 1 hour with no tool calls), auto-cancel and notify the user.
- **Agent tries to bypass restrictions:** Tool filtering happens at the registry level before schemas are sent to the LLM. The LLM never sees blocked tools, so it can't call them. If it hallucinates a tool call, the tool loop's "unknown tool" handling rejects it.

## Dependencies

- **006-execution-threads**: Planning mode interacts with thread management (main thread continues during planning).
- **007-context-management**: Plan is included in context during execution phase.
- **008-llm-integration**: Tool loop modifications for tool filtering and planning tools.
- **010-agent-self-edit**: Self-edit tools must be blockable during planning mode.
