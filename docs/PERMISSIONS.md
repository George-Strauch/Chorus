# Permission Profiles

This document specifies the permission system used by Chorus to control what actions agents can perform. The system is based on regex pattern matching against action strings, using an allow/ask/deny model.

---

## Overview

Every tool invocation in Chorus produces an **action string** that describes what the agent is trying to do. This string is matched against the agent's **permission profile** to determine whether the action should be allowed automatically, require human confirmation, or be denied outright.

The permission engine is the gatekeeper between the LLM's intent (tool calls) and actual execution. It runs **before** any tool code executes.

---

## Action String Format

Action strings follow the format:

```
tool:<tool_name>:<detail>
```

Where:
- `tool:` is a fixed prefix indicating this is a tool action
- `<tool_name>` is the registered name of the tool (e.g., `bash`, `create_file`, `git_push`)
- `<detail>` provides context about the specific invocation

The detail portion varies by tool and is constructed from the tool's parameters. It is designed to give the permission engine enough information to make fine-grained decisions.

### Action Strings by Tool

| Tool | Action String | Detail Contents |
|---|---|---|
| `create_file` | `tool:create_file:<path>` | Relative file path within workspace |
| `str_replace` | `tool:str_replace:<path>` | Relative file path being edited |
| `view` | `tool:view:<path>` | Relative file path being read |
| `bash` | `tool:bash:<command>` | The full command string |
| `git_init` | `tool:git:init` | Fixed string |
| `git_commit` | `tool:git:commit` | Fixed string |
| `git_push` | `tool:git:push <remote> <branch>` | Remote and branch names |
| `git_branch` | `tool:git:branch <name>` | Branch name |
| `git_merge_request` | `tool:git:merge_request <target>` | Target branch |
| `self_edit_system_prompt` | `tool:self_edit:system_prompt` | Fixed string |
| `self_edit_docs` | `tool:self_edit:docs:<path>` | Doc file path |
| `self_edit_permissions` | `tool:self_edit:permissions:<profile>` | Target profile name |
| `self_edit_model` | `tool:self_edit:model:<model_name>` | Target model name |

### Examples

```
tool:create_file:src/main.py
tool:str_replace:config/settings.yaml
tool:view:README.md
tool:bash:npm install
tool:bash:rm -rf node_modules
tool:bash:curl https://example.com/api
tool:git:push origin main
tool:git:push origin feature/auth
tool:git:branch feature/new-ui
tool:git:merge_request main
tool:self_edit:system_prompt
tool:self_edit:docs:README.md
tool:self_edit:permissions:open
tool:self_edit:model:claude-sonnet-4-20250514
```

---

## Permission Profile Structure

A permission profile is a pair of ordered regex pattern lists:

```python
@dataclass
class PermissionProfile:
    allow: list[str]  # Regex patterns -- matched actions proceed automatically
    ask: list[str]    # Regex patterns -- matched actions require human confirmation
```

Both `allow` and `ask` contain regular expression patterns that are matched against the full action string. Patterns use Python's `re` module syntax and are matched with `re.fullmatch()` -- the pattern must match the entire action string, not just a substring.

---

## Matching Order

When the permission engine receives an action string, it evaluates it in strict order:

```
1. Check ALLOW list:
   - Iterate through allow patterns in order
   - If any pattern matches the action string -> ALLOW (execute immediately)

2. Check ASK list:
   - Iterate through ask patterns in order
   - If any pattern matches the action string -> ASK (prompt user for confirmation)

3. Implicit DENY:
   - If no pattern in either list matches -> DENY (reject with error message)
```

This means:
- **Allow takes priority over ask.** If an action matches both an allow pattern and an ask pattern, it is allowed without confirmation.
- **Deny is the default.** Any action that does not match at least one pattern in allow or ask is silently denied.
- **Order within a list does not matter for the decision** (any match in the list triggers the decision), but patterns are evaluated in order and short-circuit on first match for performance.

---

## Built-in Presets

Chorus ships with three built-in permission presets. Agents are assigned one by name (stored in `agent.json`). Custom profiles can also be defined.

### `open` -- Unrestricted

Allows all actions without confirmation. Suitable for fully trusted agents on private servers where you want maximum autonomy.

```python
PermissionProfile(
    allow=[
        "tool:.*"    # Match any action string starting with "tool:"
    ],
    ask=[]           # Nothing requires confirmation
)
```

This single pattern `tool:.*` matches every possible action string because all action strings start with `tool:` and `.*` matches any remaining characters.

### `standard` -- Balanced (Default)

Automatically allows file operations and read-only actions. Requires confirmation for bash commands, git push/merge operations, and self-edit actions. This is the default profile assigned to new agents.

```python
PermissionProfile(
    allow=[
        "tool:create_file:.*",        # Create/overwrite any file in workspace
        "tool:str_replace:.*",         # Edit any file in workspace
        "tool:view:.*",                # Read any file
        "tool:git:init",               # Initialize git repo
        "tool:git:commit",             # Commit changes
        "tool:git:branch .*",          # Create branches
    ],
    ask=[
        "tool:bash:.*",                # All bash commands require confirmation
        "tool:git:push .*",            # Push requires confirmation
        "tool:git:merge_request .*",   # MR creation requires confirmation
        "tool:self_edit:.*",           # All self-modifications require confirmation
    ]
)
```

With this profile:
- File creation, editing, and reading happen silently
- Git commits and branching happen silently
- Every bash command shows a confirmation prompt in Discord
- Pushing code and creating merge requests require approval
- Any self-edit (prompt, docs, permissions, model) requires approval
- Any action not listed (if new tools are added) is denied by default

### `locked` -- Read-Only

Only allows viewing files. Everything else is denied. Useful for agents that should only observe and respond, or for temporarily restricting an agent.

```python
PermissionProfile(
    allow=[
        "tool:view:.*"               # Can read any file
    ],
    ask=[]                            # Nothing goes to ask -- unlisted actions are denied
)
```

With this profile:
- The agent can read files
- All writes, bash commands, git operations, and self-edits are denied
- The agent can still respond to messages (LLM responses are not gated by permissions, only tool use is)

---

## Custom Profiles

Custom permission profiles can be defined in the server-wide `config.json` or in an individual agent's `agent.json`. A custom profile specifies the full allow and ask lists:

```json
{
    "permissions": {
        "allow": [
            "tool:view:.*",
            "tool:create_file:docs/.*",
            "tool:str_replace:docs/.*",
            "tool:self_edit:docs:.*"
        ],
        "ask": [
            "tool:bash:.*",
            "tool:create_file:.*",
            "tool:str_replace:.*"
        ]
    }
}
```

This example profile:
- Automatically allows reading any file and editing files within `docs/`
- Automatically allows the agent to edit its own documentation
- Requires confirmation for bash commands and file edits outside `docs/`
- Denies all git operations and self-edits to system prompt, permissions, or model

### Custom Profile Design Guidelines

When designing custom profiles, consider:

1. **Start restrictive, add permissions.** Begin with `locked` and add patterns for specific actions the agent needs.

2. **Use specific detail patterns.** Instead of `tool:bash:.*` in the allow list, consider patterns like `tool:bash:npm .*` or `tool:bash:python .*` to restrict which commands run automatically.

3. **Use ask as a safety net.** Put broad patterns in `ask` to catch actions you did not anticipate, rather than letting them fall through to deny:

    ```python
    allow=["tool:view:.*", "tool:create_file:src/.*"],
    ask=["tool:.*"]  # Catch-all: anything not explicitly allowed gets a prompt
    ```

4. **Be aware of regex anchoring.** Patterns are matched with `re.fullmatch()`, so `tool:bash:ls` matches only exactly `tool:bash:ls`, not `tool:bash:ls -la`. Use `tool:bash:ls.*` to match all `ls` variations.

5. **Escape special regex characters.** If you need to match literal dots, brackets, or other regex metacharacters in file paths, escape them: `tool:create_file:src/config\.json` to match only that specific file.

---

## Pattern Matching Details

### Regex Engine

Patterns use Python's `re` module. Key behaviors:

- `re.fullmatch()` is used, meaning the pattern must match the **entire** action string
- Patterns are compiled once when the profile is loaded, not on every check
- Invalid regex patterns cause the profile to fail validation at load time

### Common Pattern Examples

```python
# Match a specific tool entirely
"tool:view:.*"                      # Any view action

# Match a specific file path
"tool:create_file:src/main\.py"     # Only this exact file

# Match files in a directory
"tool:create_file:src/.*"           # Any file under src/
"tool:str_replace:tests/.*\.py"     # Python files under tests/

# Match specific bash commands
"tool:bash:npm .*"                  # npm commands
"tool:bash:python .*"              # python commands
"tool:bash:(ls|cat|head|tail).*"    # Read-only shell commands
"tool:bash:git (status|log|diff).*" # Read-only git commands via bash

# Match git operations to specific branches
"tool:git:push origin (dev|staging)"  # Push only to dev or staging
"tool:git:push origin (?!main).*"     # Push to any branch EXCEPT main

# Match all self-edit actions
"tool:self_edit:.*"                 # Any self-edit

# Match specific self-edits
"tool:self_edit:docs:.*"            # Only doc edits
"tool:self_edit:(system_prompt|model:.*)"  # Prompt and model changes

# Catch-all
"tool:.*"                           # Any tool action
```

---

## Role-Based Gating for Self-Edit Permissions

When an agent attempts to edit its own permission profile via `self_edit_permissions`, an additional check occurs beyond the normal permission engine flow. The system verifies that the **Discord user whose message triggered the current agent session** has a role that authorizes granting the requested permission level.

### How It Works

1. Agent calls `self_edit_permissions("open")` during a tool loop
2. Permission engine checks the action string `tool:self_edit:permissions:open` -- if it is denied or the user declines the ask prompt, the action stops here
3. If the permission engine allows it (or the user approves the ask), a **role check** occurs:
   - The system looks up the Discord user who sent the message that started this tool loop iteration
   - It checks that user's Discord roles against a role-permission mapping configured in server settings
   - If the user has a role that allows granting the `open` profile, the self-edit proceeds
   - If not, the action is denied with an error message explaining insufficient role privileges

### Role-Permission Mapping

Configured in server-wide settings:

```json
{
    "permission_grants": {
        "admin": ["open", "standard", "locked"],
        "developer": ["standard", "locked"],
        "viewer": ["locked"]
    }
}
```

This means:
- Users with the `admin` Discord role can grant any built-in profile
- Users with the `developer` role can grant `standard` or `locked` but not `open`
- Users with the `viewer` role can only grant `locked`
- Custom profiles follow additional configuration rules

If no role mapping is configured, the default behavior is: only server administrators (users with the `Administrator` Discord permission) can change agent permission profiles.

---

## ASK Confirmation Flow

When an action matches an `ask` pattern, the system posts a confirmation prompt in the Discord channel:

```
Agent wants to execute:
  bash: rm -rf node_modules && npm install

Action: tool:bash:rm -rf node_modules && npm install

React with checkmark to approve or X to deny.
```

The system waits for the user to react or reply. The timeout for confirmation is configurable (default: 60 seconds). If the timeout expires, the action is denied.

Only users with appropriate roles can approve ask prompts. The specific role requirements are configurable per server.

### ASK Behavior Details

- While waiting for confirmation, the tool loop is paused. The LLM is not called again until the pending action is resolved.
- If the user denies the action, the tool loop sends a `ToolResult` with an error message back to the LLM, which can then decide to try a different approach or report the denial.
- Multiple tool calls in a single LLM response are confirmed individually in sequence. If one is denied, subsequent tool calls in that batch are still presented for confirmation.

---

## Permission Engine Implementation

The permission engine is implemented in `permissions/engine.py`. Key aspects:

```python
class PermissionEngine:
    def check(self, profile: PermissionProfile, action: str) -> Decision:
        """
        Check an action string against a permission profile.

        Returns Decision.ALLOW, Decision.ASK, or Decision.DENY.
        """
        for pattern in profile.allow_compiled:
            if pattern.fullmatch(action):
                return Decision.ALLOW

        for pattern in profile.ask_compiled:
            if pattern.fullmatch(action):
                return Decision.ASK

        return Decision.DENY

    def resolve_profile(self, profile_name_or_dict: str | dict) -> PermissionProfile:
        """
        Resolve a profile from a name (built-in preset) or a dict (custom profile).
        Compiles all regex patterns and validates them.
        """
        ...
```

### Audit Logging

Every permission check result is logged to the SQLite audit table:

```
| agent_name | timestamp | action_string | decision | user_id | detail |
|---|---|---|---|---|---|
| frontend | 2026-02-12T10:00:00 | tool:bash:npm test | allow | 12345 | |
| frontend | 2026-02-12T10:00:05 | tool:git:push origin main | ask_approved | 12345 | |
| frontend | 2026-02-12T10:00:10 | tool:bash:rm -rf / | ask_denied | 12345 | User denied |
| writing | 2026-02-12T10:01:00 | tool:bash:ls | deny | 67890 | Profile: locked |
```

This provides a complete audit trail of what every agent has done (or tried to do), which is essential for debugging and security review.

---

## Interaction with Tool Registry

The tool registry works with the permission engine in two ways:

1. **Pre-filtering tool schemas:** Before sending the tool list to the LLM, the registry checks each tool against the permission profile. If a tool would be denied for ALL possible detail values (e.g., `tool:bash:.*` matches no pattern in a `locked` profile), it is omitted from the schema sent to the LLM. This prevents the LLM from attempting actions it can never perform.

2. **Runtime checking:** Even if a tool is included in the schema, every individual invocation is still checked against the permission engine at execution time. This handles cases where a tool is partially allowed (e.g., `create_file` is allowed for `src/.*` but not for `config/.*`).

The pre-filtering is an optimization, not a security measure. The runtime check is the authoritative gate.
