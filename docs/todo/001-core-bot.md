# TODO 001 — Core Bot Skeleton

## Objective

Stand up the Discord bot foundation: connect to Discord with `discord.py`, register the slash command tree via `app_commands`, implement cog-based command loading, and wire up essential event handlers (`on_ready`, `on_message`, error handling). After this TODO, the bot starts, connects to Discord, loads cogs dynamically, and responds to a `/ping` health-check command.

## Acceptance Criteria

- Bot connects to Discord using a token from `.env` (`DISCORD_TOKEN`).
- `ChorusBot` subclass of `discord.Client` (or `commands.Bot`) initializes with `discord.Intents` configured for messages, guilds, and message content.
- Slash command tree syncs on `on_ready` (with optional guild-scoped sync for development).
- Cog loader discovers and loads all modules under `chorus/commands/` automatically.
- `/ping` slash command returns latency in ms — serves as proof-of-life.
- `on_ready` logs bot name, connected guilds, and loaded cog count.
- `on_message` ignores bot's own messages; passes non-command messages through for future agent handling.
- Global `app_commands` error handler catches `MissingPermissions`, `CommandNotFound`, sends user-friendly ephemeral responses.
- `__main__.py` entry point loads `.env`, constructs bot, calls `bot.run()`.
- `config.py` reads and validates env vars (`DISCORD_TOKEN` required; `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` optional at this stage).
- All async code uses `async/await` — no blocking calls on the event loop.
- Type hints on every function signature.

## Tests to Write First

File: `tests/test_config.py`
```
test_config_loads_discord_token_from_env
test_config_raises_on_missing_discord_token
test_config_optional_api_keys_default_to_none
test_config_strips_whitespace_from_token
test_config_rejects_empty_string_token
```

File: `tests/test_commands/test_agent_commands.py` (initial ping test only)
```
test_ping_command_returns_latency_message
```

File: `tests/test_bot.py` (new, or inline in conftest verification)
```
test_bot_creates_with_correct_intents
test_bot_loads_cogs_from_commands_package
test_bot_on_message_ignores_self
test_bot_error_handler_sends_ephemeral_on_missing_permissions
```

File: `tests/conftest.py` (fixtures)
```
# Fixtures to build:
mock_discord_ctx          — AsyncMock of discord.Interaction with channel, guild, user
mock_bot                  — ChorusBot instance with mocked connection (never actually connects)
tmp_env                   — monkeypatch env vars for config tests
```

## Implementation Notes

1. **Bot class location:** `src/chorus/bot.py`. Subclass `commands.Bot` from `discord.ext.commands` — this gives us both prefix commands (unused but available) and the `app_commands` tree in one object.

2. **Intents:** Enable `guilds`, `guild_messages`, `message_content`. Do NOT enable `presences` or `members` unless a future TODO requires them — principle of least privilege.

3. **Cog loading pattern:**
   ```python
   async def setup_hook(self) -> None:
       for module in pkgutil.iter_modules(chorus.commands.__path__):
           await self.load_extension(f"chorus.commands.{module.name}")
   ```
   This auto-discovers cogs without a hardcoded list.

4. **Config module:** Use `python-dotenv` to load `.env`. Expose a frozen dataclass:
   ```python
   @dataclass(frozen=True)
   class BotConfig:
       discord_token: str
       anthropic_api_key: str | None = None
       openai_api_key: str | None = None
       chorus_home: Path = Path.home() / ".chorus-agents"
       dev_guild_id: int | None = None  # For guild-scoped command sync during dev
   ```
   Validate at construction time — fail fast if `DISCORD_TOKEN` is missing or empty.

5. **Command sync strategy:** In `on_ready`, sync globally by default. If `DEV_GUILD_ID` is set, also sync to that guild (guild sync is instant; global sync takes up to an hour). Log which mode was used.

6. **Error handler:** Register `bot.tree.on_error` to catch `app_commands.errors`. Convert exceptions to short ephemeral messages. Log full tracebacks server-side.

7. **Logging:** Use `logging.getLogger("chorus")`. Configure in `__main__.py` with `logging.basicConfig(level=logging.INFO)`. All modules use child loggers (`chorus.bot`, `chorus.config`, etc.).

8. **Testing without Discord:** All tests mock the Discord connection. Use `unittest.mock.AsyncMock` for the bot's websocket. The bot object should be constructible without a live token for testing — guard `bot.run()` behind the `__main__` entry point.

## Dependencies

None. This is the foundation.
