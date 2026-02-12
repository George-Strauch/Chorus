# TODO 010 — Agent Self-Edit

## Objective

Implement the agent's ability to edit its own configuration: system prompt, docs, permission profile, and model. Self-editing is a first-class capability — agents should be able to evolve their own behavior as they work. All self-edits are gated by the invoking user's Discord role and logged to the audit table.

## Acceptance Criteria

- `self_edit_system_prompt(agent, new_prompt)` updates the agent's system prompt in `agent.json`. Takes effect on the next LLM call (no restart needed).
- `self_edit_docs(agent, path, content)` creates or updates a file in the agent's `docs/` directory. Path traversal prevention applies (confined to `docs/`).
- `self_edit_permissions(agent, new_profile, invoking_user_roles)` updates the agent's permission profile. Gated: the invoking user must have a Discord role that is allowed to grant the requested permission level (e.g., only admins can set `open` permissions).
- `self_edit_model(agent, new_model)` switches the agent's model. Validates against `available_models.json`.
- All self-edit operations are exposed as tools in the tool registry so the LLM can invoke them during the tool loop.
- Permission action strings: `tool:self_edit:system_prompt`, `tool:self_edit:docs <path>`, `tool:self_edit:permissions <profile>`, `tool:self_edit:model <model>`.
- Every self-edit is logged in the SQLite audit table with: agent name, edit type, old value (truncated), new value (truncated), invoking user, timestamp.
- The agent's `docs/` directory is always included in the system prompt context. Edits to docs directly influence future agent behavior.
- Self-edit of permissions has a role hierarchy check:
  - Any user can set `locked` or `standard`.
  - Only users with `Manage Server` permission can set `open` or custom profiles.
  - The agent itself cannot escalate beyond what the invoking user is allowed to grant.

## Tests to Write First

File: `tests/test_agent/test_self_edit.py`
```
# System prompt
test_self_edit_system_prompt_updates_agent_json
test_self_edit_system_prompt_old_value_logged
test_self_edit_system_prompt_permission_check
test_self_edit_system_prompt_empty_string_rejected

# Docs
test_self_edit_docs_creates_new_file
test_self_edit_docs_updates_existing_file
test_self_edit_docs_path_traversal_prevented
test_self_edit_docs_subdirectory_creation
test_self_edit_docs_permission_check

# Permissions
test_self_edit_permissions_to_standard
test_self_edit_permissions_to_locked
test_self_edit_permissions_to_open_requires_admin
test_self_edit_permissions_custom_profile_requires_admin
test_self_edit_permissions_non_admin_cannot_set_open
test_self_edit_permissions_role_check_uses_invoking_user
test_self_edit_permissions_old_profile_logged

# Model
test_self_edit_model_updates_agent_json
test_self_edit_model_validates_against_available
test_self_edit_model_rejects_unavailable
test_self_edit_model_permission_check

# Audit logging
test_self_edit_logs_to_audit_table
test_audit_log_contains_agent_name
test_audit_log_contains_old_and_new_values
test_audit_log_truncates_long_values
test_audit_log_contains_timestamp

# Tool integration
test_self_edit_tools_registered_in_registry
test_self_edit_tool_schema_correct
test_self_edit_tool_invokable_from_tool_loop
```

## Implementation Notes

1. **Module location:** `src/chorus/agent/self_edit.py`.

2. **Core functions:**
   ```python
   async def edit_system_prompt(
       agent_dir: Path, new_prompt: str, user_id: int, db: Database
   ) -> SelfEditResult: ...

   async def edit_docs(
       agent_dir: Path, doc_path: str, content: str, user_id: int, db: Database
   ) -> SelfEditResult: ...

   async def edit_permissions(
       agent_dir: Path, new_profile: str | dict, user_roles: list[str],
       user_id: int, db: Database
   ) -> SelfEditResult: ...

   async def edit_model(
       agent_dir: Path, new_model: str, available_models: list[str],
       user_id: int, db: Database
   ) -> SelfEditResult: ...
   ```

3. **SelfEditResult dataclass:**
   ```python
   @dataclass
   class SelfEditResult:
       success: bool
       edit_type: str
       old_value: str
       new_value: str
       message: str
       error: str | None = None
   ```

4. **Role hierarchy for permission editing:** Define a mapping of permission levels to required Discord permissions:
   ```python
   PERMISSION_LEVEL_REQUIREMENTS = {
       "locked": None,          # Anyone can set locked
       "standard": None,        # Anyone can set standard
       "open": "manage_guild",  # Only admins
   }
   ```
   Custom profiles (dict-based) always require `manage_guild`. Check `interaction.user.guild_permissions.manage_guild`.

5. **Audit table schema:**
   ```sql
   CREATE TABLE audit_log (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       timestamp TEXT NOT NULL,
       agent_name TEXT NOT NULL,
       edit_type TEXT NOT NULL,      -- "system_prompt", "docs", "permissions", "model"
       old_value TEXT,               -- Truncated to 500 chars
       new_value TEXT,               -- Truncated to 500 chars
       user_id INTEGER NOT NULL,
       user_name TEXT
   );
   ```

6. **Docs directory context injection:** When building the system prompt for LLM calls, append the contents of all files in `docs/` as a section:
   ```
   <agent_docs>
   # docs/README.md
   [content]

   # docs/architecture.md
   [content]
   </agent_docs>
   ```
   This means edits to `docs/` immediately affect the agent's behavior on the next turn. The context module (007) or tool loop (008) should handle this injection — `self_edit.py` just writes the files.

7. **Path traversal for docs:** Reuse the `resolve_in_workspace` utility from TODO 004, but with `docs/` as the root instead of `workspace/`. The agent should NOT be able to write to `workspace/` via self_edit_docs — that's what the file tools are for.

8. **Atomic writes:** When updating `agent.json`, read-modify-write with a temporary file + rename to prevent corruption on crash:
   ```python
   tmp = agent_json_path.with_suffix(".tmp")
   tmp.write_text(json.dumps(data, indent=2))
   tmp.rename(agent_json_path)
   ```

9. **Tool registration:** Register four self-edit tools in the registry:
   - `self_edit_system_prompt` — params: `new_prompt: str`
   - `self_edit_docs` — params: `path: str, content: str`
   - `self_edit_permissions` — params: `profile: str | dict`
   - `self_edit_model` — params: `model: str`

## Dependencies

- **002-agent-manager**: Agent directory structure and `agent.json` management.
- **003-permission-profiles**: Permission checks for self-edit operations, and profile validation for permission editing.
- **004-file-tools**: Path traversal prevention utilities (reused for docs jail).
