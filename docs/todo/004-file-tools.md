# TODO 004 — File Tools

## Objective

Implement the file operation tools that agents use to interact with their workspace: `create_file`, `str_replace`, and `view`. All operations are confined to the agent's `workspace/` directory with mandatory path traversal prevention. Every tool invocation runs through the permission engine before execution.

## Acceptance Criteria

- `create_file(path, content)` creates a file (and any intermediate directories) within the agent's workspace. Returns the absolute path of the created file.
- `str_replace(path, old_str, new_str)` performs an exact string replacement in a file. Fails if `old_str` is not found or is not unique (appears more than once). Returns the updated file content around the edit.
- `view(path, offset?, limit?)` returns the content of a file with line numbers. Supports optional `offset` (line number to start from) and `limit` (number of lines to return). Defaults to the full file.
- All paths are resolved relative to the agent's workspace root (`~/.chorus-agents/agents/<name>/workspace/`).
- **Path traversal prevention:** Before any I/O, the resolved absolute path is checked to be within the workspace root. Paths like `../../etc/passwd`, `/etc/passwd`, or symlinks pointing outside are rejected with `PathTraversalError`.
- Symlink resolution: `Path.resolve()` is called BEFORE the jail check so symlinks can't escape.
- Permission checks happen before execution: `tool:file:create <path>`, `tool:file:str_replace <path>`, `tool:file:view <path>`.
- Tools return structured results (dataclass or dict) suitable for feeding back into the LLM tool loop.
- File encoding is UTF-8. Binary files are rejected with a clear error message.
- `create_file` overwrites if the file already exists (like `Write` in Claude Code).
- `str_replace` includes context lines (3 before and after the edit) in its return value for LLM context.
- All tools are registered in `tools/registry.py` with metadata (name, description, parameter schema).

## Tests to Write First

File: `tests/test_tools/test_file_ops.py`
```
# create_file
test_create_file_in_workspace_root
test_create_file_in_subdirectory_creates_parents
test_create_file_overwrites_existing
test_create_file_rejects_path_traversal_dotdot
test_create_file_rejects_absolute_path_outside_workspace
test_create_file_rejects_symlink_escape
test_create_file_permission_check_runs_before_write
test_create_file_utf8_content

# str_replace
test_str_replace_exact_match
test_str_replace_fails_on_no_match
test_str_replace_fails_on_multiple_matches
test_str_replace_returns_context_lines
test_str_replace_rejects_path_traversal
test_str_replace_permission_check_runs_before_edit
test_str_replace_preserves_file_permissions
test_str_replace_handles_empty_new_string_for_deletion

# view
test_view_returns_full_file_with_line_numbers
test_view_with_offset_and_limit
test_view_offset_beyond_file_length
test_view_rejects_path_traversal
test_view_rejects_binary_file
test_view_permission_check_runs_before_read
test_view_empty_file
test_view_nonexistent_file_raises

# Path traversal (dedicated tests)
test_jail_check_allows_nested_paths
test_jail_check_blocks_dotdot
test_jail_check_blocks_absolute_outside
test_jail_check_resolves_symlinks_before_checking
test_jail_check_blocks_double_encoded_traversal
```

File: `tests/test_tools/test_registry.py`
```
test_registry_contains_file_tools
test_registry_tool_has_name_and_description
test_registry_tool_has_parameter_schema
test_registry_get_tool_by_name
test_registry_list_all_tools
```

File: `tests/conftest.py` (new fixtures)
```
workspace_dir    — tmp_path with a few pre-populated files for testing
agent_with_workspace — full agent dir structure with workspace pre-populated
```

## Implementation Notes

1. **Module location:** `src/chorus/tools/file_ops.py`. Import `PermissionProfile` and `check` from `chorus.permissions.engine`.

2. **Path jail function** — extract to a shared utility since `bash.py` and `git.py` will also need it:
   ```python
   # src/chorus/tools/_path_jail.py or within file_ops.py
   def resolve_in_workspace(workspace: Path, relative_path: str) -> Path:
       """Resolve a path within workspace, raising PathTraversalError if it escapes."""
       resolved = (workspace / relative_path).resolve()
       workspace_resolved = workspace.resolve()
       if not str(resolved).startswith(str(workspace_resolved) + "/") and resolved != workspace_resolved:
           raise PathTraversalError(f"Path {relative_path!r} resolves outside workspace")
       return resolved
   ```
   Use `os.path.commonpath` or the string prefix check above. The string check must include the trailing `/` to prevent `/workspace-evil` matching `/workspace`.

3. **create_file implementation:**
   ```python
   async def create_file(workspace: Path, path: str, content: str, profile: PermissionProfile) -> FileResult:
       action = format_action("file", f"create {path}")
       result = check(action, profile)
       if result == PermissionResult.DENY:
           raise PermissionDeniedError(action)
       # ASK result is handled by the caller (tool loop) — return ASK status
       resolved = resolve_in_workspace(workspace, path)
       resolved.parent.mkdir(parents=True, exist_ok=True)
       resolved.write_text(content, encoding="utf-8")
       return FileResult(path=str(resolved), action="created", ...)
   ```

4. **str_replace uniqueness check:** Read the file, use `content.count(old_str)` to check for uniqueness. If count is 0, raise `StringNotFoundError`. If count > 1, raise `AmbiguousMatchError` with the count. Then do a single `content.replace(old_str, new_str, 1)`.

5. **Binary file detection for view:** Read the first 8192 bytes and check for null bytes. If `b"\x00" in chunk`, raise `BinaryFileError`.

6. **Tool registry pattern** (`src/chorus/tools/registry.py`):
   ```python
   @dataclass
   class ToolDefinition:
       name: str
       description: str
       parameters: dict[str, Any]  # JSON Schema
       handler: Callable[..., Awaitable[Any]]

   class ToolRegistry:
       def __init__(self) -> None:
           self._tools: dict[str, ToolDefinition] = {}
       def register(self, tool: ToolDefinition) -> None: ...
       def get(self, name: str) -> ToolDefinition | None: ...
       def list_all(self) -> list[ToolDefinition]: ...
       def get_for_provider(self, provider: str) -> list[dict]: ...  # Format for Anthropic/OpenAI tool schemas
   ```
   File tools register themselves at module import time or via an explicit `register_file_tools(registry)` function.

7. **Return types:** Define `FileResult` dataclass with fields: `path`, `action`, `content_snippet`, `success`, `error`. The LLM tool loop will serialize this back into the conversation.

## Dependencies

- **002-agent-manager**: Agent workspace directory must exist and be managed.
- **003-permission-profiles**: Permission checks must be functional.
