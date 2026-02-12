# TODO 009 — Default Settings

> **Status:** PENDING

## Objective

Implement server-wide default settings: default model, default permission profile, and other global configuration. These defaults apply to newly created agents unless overridden. Provide slash commands for viewing and modifying settings, validating API keys, and discovering available models.

## Acceptance Criteria

- Server-wide defaults are stored in `~/.chorus-agents/config.json`.
- Default settings include: `default_model`, `default_permissions`, `idle_timeout`, `max_tool_loop_iterations`, `max_bash_timeout`.
- `/settings model <model_name>` sets the default model. Validates that the model is in `available_models.json`.
- `/settings permissions <profile_name>` sets the default permission profile. Validates against known presets or accepts a JSON object for custom profiles.
- `/settings show` displays all current settings in a formatted embed.
- `/settings validate-keys` re-runs key validation and model discovery, updates `available_models.json`, and reports results.
- Settings are loaded on bot startup and cached in memory. Changes are written through to disk immediately.
- New agents created after a default change inherit the new defaults (existing agents are not retroactively changed).
- `config.json` is created with sensible defaults on first bot startup if it doesn't exist.
- Settings commands are restricted to users with the `Manage Server` Discord permission.

## Tests to Write First

File: `tests/test_config.py` (extend from 001)
```
test_global_config_loads_from_json
test_global_config_creates_default_on_missing
test_global_config_default_model_is_none_until_set
test_global_config_default_permissions_is_standard
test_global_config_write_through_persists
test_global_config_validates_model_against_available
test_global_config_rejects_unknown_model
test_global_config_idle_timeout_default
test_global_config_serialization_roundtrip
```

File: `tests/test_commands/test_settings_commands.py`
```
test_settings_model_command_updates_default
test_settings_model_command_rejects_unavailable_model
test_settings_model_command_requires_manage_server
test_settings_permissions_command_updates_default
test_settings_permissions_command_accepts_preset_name
test_settings_permissions_command_rejects_invalid_preset
test_settings_show_command_returns_embed
test_settings_show_displays_all_fields
test_settings_validate_keys_triggers_discovery
test_settings_validate_keys_reports_results
test_settings_commands_ephemeral_responses
```

File: `tests/test_llm/test_discovery.py` (extend from 008)
```
test_validate_keys_command_updates_cache_file
test_validate_keys_reports_per_provider_status
```

## Implementation Notes

1. **Module location:** Settings management in `src/chorus/config.py` (extend the existing module), commands in `src/chorus/commands/settings_commands.py`.

2. **GlobalConfig dataclass:**
   ```python
   @dataclass
   class GlobalConfig:
       default_model: str | None = None
       default_permissions: str = "standard"
       idle_timeout: int = 1800  # 30 minutes in seconds
       max_tool_loop_iterations: int = 25
       max_bash_timeout: int = 120  # seconds

       @classmethod
       def load(cls, path: Path) -> "GlobalConfig": ...
       def save(self, path: Path) -> None: ...
   ```

3. **Settings file location:** `~/.chorus-agents/config.json`. Distinct from the bot's `.env` (which has secrets). This file holds operational defaults.

4. **Model validation flow:**
   ```
   User: /settings model claude-sonnet-4-20250514
   Bot: Reads available_models.json
        → Check if "claude-sonnet-4-20250514" is in any provider's model list
        → If yes: update config.json, respond "Default model set to claude-sonnet-4-20250514"
        → If no: respond "Model not available. Run /settings validate-keys to refresh. Available: ..."
   ```

5. **Permission to change settings:** Use `@app_commands.checks.has_permissions(manage_guild=True)`. This restricts settings changes to server admins/moderators.

6. **Settings embed format:**
   ```
   Chorus Settings
   ─────────────────────
   Default Model:       claude-sonnet-4-20250514
   Default Permissions: standard
   Idle Timeout:        30 minutes
   Max Tool Iterations: 25
   Max Bash Timeout:    120 seconds
   ─────────────────────
   API Keys:
     Anthropic: Valid (5 models)
     OpenAI:    Valid (12 models)
   ─────────────────────
   Last key validation: 2026-02-12 10:00 UTC
   ```

7. **Validate-keys flow:**
   - Call `validate_key` and `discover_models` for each provider that has a key configured.
   - Update `available_models.json`.
   - Report results as an embed with per-provider status.
   - This is an async operation — show a "Validating..." message, then edit it with results.

8. **Write-through pattern:** `GlobalConfig` holds an internal `_path` reference. On any setter, also call `self.save()`. Alternatively, use a context manager or explicit `commit()` method. Prefer explicit over magic.

9. **Default config.json:**
   ```json
   {
     "default_model": null,
     "default_permissions": "standard",
     "idle_timeout": 1800,
     "max_tool_loop_iterations": 25,
     "max_bash_timeout": 120
   }
   ```

## Dependencies

- **001-core-bot**: Bot and slash command framework.
- **008-llm-integration**: Model discovery and key validation functions.
