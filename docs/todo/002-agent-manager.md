# TODO 002 — Agent Manager

## Objective

Implement the agent lifecycle: create new agents from `template/`, set up their directory structure under `~/.chorus-agents/agents/<name>/`, destroy agents (with optional file retention), list all agents, and update agent configuration. After this TODO, the `/agent init`, `/agent destroy`, `/agent list`, and `/agent config` slash commands work end-to-end (minus LLM and tool execution, which come later).

## Acceptance Criteria

- `/agent init <name>` copies `template/` to `~/.chorus-agents/agents/<name>/`, initializes a git repo in `workspace/`, creates a Discord channel in the server, writes `channel_id` and `name` and `created_at` into `agent.json`, and registers the agent in SQLite.
- Agent names are validated: lowercase alphanumeric plus hyphens, 2-32 chars, no leading/trailing hyphens.
- `/agent init` accepts optional `system_prompt`, `model`, and `permission_profile` parameters that override the template defaults in `agent.json`.
- `/agent destroy <name>` deletes the Discord channel, removes the agent directory (or moves to a `.trash/` folder if `--keep-files` is passed), and removes the agent record from SQLite.
- `/agent destroy` requires confirmation — the command sends an ephemeral "Are you sure?" with a confirm button; no action on timeout.
- `/agent list` returns an embed listing all agents with their name, channel mention, model, and permission profile.
- `/agent config <name> <key> <value>` updates a top-level key in `agent.json`. Validates that the key is one of the allowed mutable fields (`system_prompt`, `model`, `permissions`).
- `AgentManager` class is the single source of truth for agent CRUD — commands delegate to it, never manipulate files directly.
- `directory.py` provides `AgentDirectory` with methods: `create(name) -> Path`, `destroy(name, keep_files: bool)`, `get(name) -> Path | None`, `list_all() -> list[str]`, `ensure_home()`.
- `~/.chorus-agents/` is created on first use if it doesn't exist (including `agents/`, `db/` subdirs).
- The `models.py` `Agent` dataclass is fully defined: `name`, `channel_id`, `model`, `system_prompt`, `permissions`, `created_at`, `running_tasks`.
- Path to template directory is resolved relative to the package root, not CWD.

## Tests to Write First

File: `tests/test_models.py`
```
test_agent_model_from_json_dict
test_agent_model_to_json_dict
test_agent_model_defaults
test_agent_model_rejects_invalid_name_uppercase
test_agent_model_rejects_invalid_name_special_chars
test_agent_model_rejects_name_too_short
test_agent_model_rejects_name_too_long
test_agent_model_accepts_valid_hyphenated_name
```

File: `tests/test_agent/test_directory.py`
```
test_ensure_home_creates_directory_structure
test_create_copies_template_to_agent_dir
test_create_initializes_git_repo_in_workspace
test_create_raises_if_agent_already_exists
test_destroy_removes_agent_directory
test_destroy_keep_files_moves_to_trash
test_get_returns_path_for_existing_agent
test_get_returns_none_for_missing_agent
test_list_all_returns_agent_names
test_list_all_returns_empty_when_no_agents
test_agent_json_written_with_name_and_timestamp
```

File: `tests/test_agent/test_manager.py`
```
test_manager_create_agent_full_lifecycle
test_manager_create_agent_with_overrides
test_manager_create_agent_rejects_duplicate_name
test_manager_destroy_agent_cleans_up
test_manager_destroy_agent_keep_files
test_manager_list_agents_returns_all
test_manager_configure_agent_updates_json
test_manager_configure_rejects_invalid_key
test_manager_configure_rejects_nonexistent_agent
```

File: `tests/test_commands/test_agent_commands.py`
```
test_agent_init_command_creates_agent_and_channel
test_agent_init_command_validates_name
test_agent_destroy_command_requires_confirmation
test_agent_list_command_returns_embed
test_agent_config_command_updates_value
```

File: `tests/conftest.py` (new fixtures)
```
tmp_chorus_home    — tmp_path based, monkeypatched as CHORUS_HOME
tmp_template_dir   — copy of template/ in tmp_path for isolation
mock_agent_dir     — pre-populated agent directory for tests that need an existing agent
```

## Implementation Notes

1. **AgentDirectory class** (`src/chorus/agent/directory.py`): Stateless utility class. All methods take a `chorus_home: Path` parameter (from config) so tests can inject a tmp_path. Key methods:
   ```python
   class AgentDirectory:
       def __init__(self, chorus_home: Path, template_dir: Path) -> None: ...
       def ensure_home(self) -> None: ...
       def create(self, name: str, overrides: dict[str, Any] | None = None) -> Path: ...
       def destroy(self, name: str, keep_files: bool = False) -> None: ...
       def get(self, name: str) -> Path | None: ...
       def list_all(self) -> list[str]: ...
   ```

2. **Template resolution:** The template directory path should be discovered via `importlib.resources` or `Path(__file__).resolve().parent.parent.parent / "template"`. Do NOT rely on `os.getcwd()`. Store the resolved path in `BotConfig`.

3. **Directory copy:** Use `shutil.copytree(template_dir, agent_dir)`. After copy, run `git init` in `agent_dir / "workspace"` using `asyncio.create_subprocess_exec`.

4. **Agent name validation regex:** `^[a-z0-9][a-z0-9-]{0,30}[a-z0-9]$` — this enforces lowercase, no leading/trailing hyphens, 2-32 chars.

5. **Discord channel creation:** `guild.create_text_channel(name)` returns the channel. Store `channel.id` in `agent.json`. The channel should be created in a "Chorus Agents" category — create the category if it doesn't exist.

6. **agent.json read/write:** Use `json.loads` / `json.dumps` with `indent=2`. Define helper functions `read_agent_json(path: Path) -> dict` and `write_agent_json(path: Path, data: dict) -> None` in `directory.py`.

7. **SQLite registration:** For now, just insert a row into an `agents` table with `(name, channel_id, created_at)`. The full SQLite schema comes in later TODOs, but the table should be created here. Use `aiosqlite`.

8. **Destroy flow:** The slash command should use `discord.ui.View` with a confirm/cancel button. Set a 30-second timeout. On confirm, call `AgentManager.destroy()`. On timeout or cancel, send "Cancelled."

9. **Error handling:** `AgentManager` methods raise custom exceptions (`AgentExistsError`, `AgentNotFoundError`, `InvalidAgentNameError`). The command layer catches these and sends user-friendly messages.

## Dependencies

- **001-core-bot**: Bot must be running with cog loading and slash command framework.
