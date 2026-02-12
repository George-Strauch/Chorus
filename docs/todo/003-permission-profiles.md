# TODO 003 — Permission Profiles

> **Status:** COMPLETED

## Objective

Implement the regex-based allow/ask permission engine. Every tool invocation is represented as an action string (`tool:<tool_name>:<detail>`). The engine matches the action against the agent's permission profile to decide: allow (proceed automatically), ask (prompt the user for confirmation in Discord), or deny (block silently with a logged reason). This module is pure logic with zero Discord or filesystem dependencies.

## Acceptance Criteria

- `PermissionProfile` dataclass holds two lists of regex pattern strings: `allow` and `ask`.
- `PermissionEngine.check(action: str, profile: PermissionProfile) -> PermissionResult` returns `ALLOW`, `ASK`, or `DENY`.
- Matching order: try `allow` patterns first (return `ALLOW` on first match), then `ask` patterns (return `ASK` on first match), then `DENY` if nothing matched.
- Action strings follow the format `tool:<tool_name>:<detail>` — e.g., `tool:bash:pip install requests`, `tool:git:push origin main`, `tool:file:create /src/app.py`, `tool:self_edit:system_prompt`.
- Three built-in presets are available:
  - `open` — allow: `[".*"]`, ask: `[]` (everything auto-approved)
  - `standard` — allow: `["tool:file:.*", "tool:git:(?!push).*"]`, ask: `["tool:bash:.*", "tool:git:push.*"]` (file ops auto, bash and git push need confirmation)
  - `locked` — allow: `["tool:file:view.*"]`, ask: `[]` (view-only, everything else denied)
- `get_preset(name: str) -> PermissionProfile` returns a preset by name; raises `UnknownPresetError` for invalid names.
- Regex patterns are compiled once and cached (not recompiled on every check).
- Invalid regex patterns in a profile raise `InvalidPermissionPatternError` at profile construction time, not at check time.
- `format_action(tool_name: str, detail: str) -> str` helper produces correctly formatted action strings.
- The engine is a pure function — no side effects, no I/O, no state. It receives the profile and action, returns the result.
- `PermissionResult` is an enum: `ALLOW`, `ASK`, `DENY`.

## Tests to Write First

File: `tests/test_permissions/test_engine.py`
```
# Core matching logic
test_check_allow_matches_first
test_check_ask_matches_when_allow_does_not
test_check_deny_when_nothing_matches
test_check_allow_takes_priority_over_ask_for_same_pattern
test_check_empty_allow_and_ask_denies_everything
test_check_full_action_string_format

# Built-in presets
test_preset_open_allows_everything
test_preset_open_allows_bash_rm
test_preset_standard_allows_file_create
test_preset_standard_allows_file_view
test_preset_standard_asks_bash
test_preset_standard_asks_git_push
test_preset_standard_allows_git_commit
test_preset_standard_denies_self_edit_system_prompt
test_preset_locked_allows_file_view
test_preset_locked_denies_file_create
test_preset_locked_denies_bash
test_preset_locked_denies_git_push
test_get_preset_returns_correct_profile
test_get_preset_raises_on_unknown_name

# Pattern specificity
test_specific_allow_overrides_broad_ask
test_regex_special_chars_in_action_handled
test_multiline_action_string_not_matched_across_lines

# Edge cases
test_action_string_with_colons_in_detail
test_empty_detail_string
test_format_action_produces_correct_string
test_invalid_regex_raises_at_construction
test_profile_compiles_patterns_once
test_profile_serialization_to_dict
test_profile_deserialization_from_dict
```

## Implementation Notes

1. **Module location:** `src/chorus/permissions/engine.py`. No imports from `chorus.agent`, `chorus.tools`, or `chorus.bot` — this module must remain dependency-free.

2. **PermissionProfile implementation:**
   ```python
   @dataclass
   class PermissionProfile:
       allow: list[str]
       ask: list[str]
       _compiled_allow: list[re.Pattern[str]] = field(init=False, repr=False)
       _compiled_ask: list[re.Pattern[str]] = field(init=False, repr=False)

       def __post_init__(self) -> None:
           try:
               self._compiled_allow = [re.compile(p) for p in self.allow]
               self._compiled_ask = [re.compile(p) for p in self.ask]
           except re.error as e:
               raise InvalidPermissionPatternError(str(e)) from e
   ```

3. **PermissionEngine as a namespace class or module-level functions:** Since there's no state, prefer module-level functions or a class with only `@staticmethod` methods. The key function:
   ```python
   def check(action: str, profile: PermissionProfile) -> PermissionResult:
       for pattern in profile._compiled_allow:
           if pattern.fullmatch(action):
               return PermissionResult.ALLOW
       for pattern in profile._compiled_ask:
           if pattern.fullmatch(action):
               return PermissionResult.ASK
       return PermissionResult.DENY
   ```
   Use `fullmatch`, not `search` or `match` — the entire action string must match the pattern. This prevents `tool:file:.*` from accidentally matching `tool:file_evil:hack` when using `search`.

4. **Preset definitions:** Store as a module-level dict:
   ```python
   PRESETS: dict[str, PermissionProfile] = {
       "open": PermissionProfile(allow=[".*"], ask=[]),
       "standard": PermissionProfile(
           allow=[r"tool:file:.*", r"tool:git:(?!push).*"],
           ask=[r"tool:bash:.*", r"tool:git:push.*"],
       ),
       "locked": PermissionProfile(allow=[r"tool:file:view.*"], ask=[]),
   }
   ```

5. **Serialization:** `PermissionProfile` needs `to_dict() -> dict` and `@classmethod from_dict(cls, data: dict) -> PermissionProfile` for JSON storage in `agent.json`. When `agent.json` has `"permissions": "standard"`, resolve to the preset. When it has `"permissions": {"allow": [...], "ask": [...]}`, parse as a custom profile.

6. **Action string format function:**
   ```python
   def format_action(tool_name: str, detail: str = "") -> str:
       return f"tool:{tool_name}:{detail}"
   ```

7. **Testing approach:** This module is ideal for pure unit tests. No mocking needed, no async, no fixtures beyond simple data construction. Parameterize heavily with `@pytest.mark.parametrize`.

---

## Discord Ask UI — `PermissionAskView`

The `ASK` result from the permission engine must have a clean, reliable Discord UI for prompting the user. This is a first-class component, not an afterthought — it is the primary way the user stays in control of what agents do.

**Location:** `src/chorus/permissions/ask_ui.py`

**Design:** A `discord.ui.View` with Allow/Deny buttons and `view.wait()` to suspend the tool loop coroutine without blocking the event loop. The view must:

1. Show the action string and human-readable context (tool name, arguments) in an embed.
2. Lock buttons to the user who triggered the agent via `interaction_check` — other server members cannot approve/deny.
3. Time out after a configurable duration (default 120s). Timeout = deny.
4. Disable buttons after a decision or timeout so stale prompts don't look interactive.

```python
class PermissionAskView(discord.ui.View):
    def __init__(self, requester_id: int, timeout: float = 120):
        super().__init__(timeout=timeout)
        self.requester_id = requester_id
        self.value: bool | None = None  # None = timed out

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Only the user who triggered the agent can respond."""
        return interaction.user.id == self.requester_id

    @discord.ui.button(label="Allow", style=discord.ButtonStyle.green)
    async def allow(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("Approved.", ephemeral=True)
        self.value = True
        self.stop()

    @discord.ui.button(label="Deny", style=discord.ButtonStyle.red)
    async def deny(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("Denied.", ephemeral=True)
        self.value = False
        self.stop()

    async def on_timeout(self) -> None:
        """Disable buttons when the view times out."""
        for child in self.children:
            child.disabled = True  # type: ignore[union-attr]
```

**The ask callback** passed to the tool loop:

```python
async def ask_user_permission(
    channel: discord.TextChannel,
    requester_id: int,
    action_string: str,
    tool_name: str,
    arguments: str,
) -> bool:
    embed = discord.Embed(
        title="Permission Required",
        description=f"`{action_string}`",
        color=discord.Color.yellow(),
    )
    embed.add_field(name="Tool", value=tool_name, inline=True)
    embed.add_field(name="Arguments", value=f"```\n{arguments}\n```", inline=False)

    view = PermissionAskView(requester_id=requester_id, timeout=120)
    msg = await channel.send(embed=embed, view=view)

    timed_out = await view.wait()

    # Disable buttons on the message after resolution
    for child in view.children:
        child.disabled = True  # type: ignore[union-attr]
    await msg.edit(view=view)

    if timed_out or view.value is None:
        return False
    return view.value
```

**Tests** (add to `tests/test_permissions/test_engine.py` or a new `tests/test_permissions/test_ask_ui.py`):
```
test_ask_view_allow_sets_value_true
test_ask_view_deny_sets_value_false
test_ask_view_timeout_leaves_value_none
test_ask_view_interaction_check_rejects_wrong_user
test_ask_view_interaction_check_allows_requester
test_ask_view_disables_buttons_on_timeout
```

## Dependencies

None. The permission engine is pure logic. The ask UI depends only on `discord.py` and is kept in a separate file from the engine so the engine remains dependency-free.
