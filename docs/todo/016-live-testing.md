# TODO 016 — Live Testing Suite

> **Status:** PENDING

## Objective

Provide a live testing framework that runs inside the deployed Discord bot, verifying real behavior (message routing, slash command latency, agent tool execution, DB connectivity, throughput). The live tests are invoked via `/test` slash commands and auto-create a dedicated test channel on startup.

## Acceptance Criteria

- Two new config fields: `live_test_enabled` (bool, default false) and `live_test_channel` (str, default `"chorus-live-tests"`), loaded from environment variables.
- On startup (if enabled), the bot finds-or-creates a test channel in a "Chorus Testing" category within the dev guild.
- `TestRunner` executes test functions, timing them and catching failures, producing `TestResult` / `SuiteResult` dataclasses.
- Named test suites (`basic`, `throughput`, `agent`, `tools`, `all`) are registered in a `SUITES` dict.
- `/test run [suite]` runs a suite and posts results as a Discord embed (green/red/yellow).
- `/test list` lists available suites and their tests.
- `/test status` shows bot uptime, latency, guild count, agent count, and test channel info.
- `LiveTestCog` is only loaded when `live_test_enabled` is true.
- Unit tests cover the runner (pass/fail/timeout/mixed) and suite registry (keys, callables).

## Dependencies

- TODO 001 (bot skeleton), TODO 002 (agent manager), TODO 005 (bash tool), TODO 004 (file tools)

## Tests to Write First

- `tests/test_testing/test_runner.py`: `test_run_test_passing`, `test_run_test_failing`, `test_run_suite_all_pass`, `test_run_suite_mixed`, `test_run_test_timeout`
- `tests/test_testing/test_suites.py`: `test_all_suites_registered`, `test_suite_functions_callable`
- `tests/test_config.py`: `test_live_test_defaults`, `test_live_test_from_env`

## Implementation Notes

- Test functions follow: `async def test_name(runner: TestRunner) -> str | None` — return detail on success, raise on failure.
- The testing module lives at `src/chorus/testing/` (separate from `tests/` which is pytest).
- The cog uses conditional loading — `setup()` checks `bot.config.live_test_enabled` and skips if false.
- Results use Discord embeds with color coding: green (all pass), red (any fail).
