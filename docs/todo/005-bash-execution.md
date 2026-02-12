# TODO 005 — Bash Execution

## Objective

Implement sandboxed bash command execution within an agent's workspace directory. Commands run as subprocesses with `cwd` set to the agent's `workspace/`, enforced timeout, output capture, and permission checks. This is the most security-sensitive tool — every design decision should favor safety.

## Acceptance Criteria

- `bash_execute(command, workspace, profile)` runs a shell command via `asyncio.create_subprocess_shell` with `cwd=workspace`.
- Permission check runs before execution: action string is `tool:bash:<command>` (the full command string).
- Configurable timeout per agent (default 120 seconds). On timeout, the process is killed (`SIGKILL` after `SIGTERM` grace period) and a timeout error is returned.
- stdout and stderr are captured separately and returned.
- Return value includes: `exit_code`, `stdout`, `stderr`, `timed_out`, `command`, `duration_ms`.
- Output is truncated to a configurable max length (default 50,000 chars) with a "[truncated]" indicator.
- The subprocess inherits a sanitized environment: `PATH`, `HOME`, `USER`, `LANG`, `TERM` only. No `DISCORD_TOKEN` or API keys leak into the subprocess.
- Multiple concurrent commands per agent are supported (agents can run background tasks).
- A command blocklist rejects known-dangerous patterns: `rm -rf /`, `:(){ :|:& };:`, `dd if=/dev/zero`, `mkfs`, `> /dev/sda`. This is a best-effort safety net, not a security boundary (Docker is the real boundary).
- The tool is registered in the tool registry with its parameter schema.
- Running processes are tracked so `/tasks` and `/status` can report them.

## Tests to Write First

File: `tests/test_tools/test_bash.py`
```
# Basic execution
test_bash_execute_simple_command
test_bash_execute_captures_stdout
test_bash_execute_captures_stderr
test_bash_execute_returns_exit_code
test_bash_execute_nonzero_exit_code
test_bash_execute_cwd_is_workspace

# Permission integration
test_bash_execute_checks_permission_before_running
test_bash_execute_denied_command_not_executed
test_bash_execute_ask_result_returned_to_caller

# Timeout
test_bash_execute_timeout_kills_process
test_bash_execute_timeout_returns_timed_out_flag
test_bash_execute_long_running_completes_within_timeout
test_bash_execute_sigterm_then_sigkill

# Output handling
test_bash_execute_truncates_long_stdout
test_bash_execute_truncates_long_stderr
test_bash_execute_empty_output
test_bash_execute_binary_output_handled

# Environment sanitization
test_bash_execute_env_does_not_contain_discord_token
test_bash_execute_env_does_not_contain_api_keys
test_bash_execute_env_contains_path
test_bash_execute_env_contains_home

# Safety
test_bash_execute_blocklist_rejects_rm_rf_root
test_bash_execute_blocklist_rejects_fork_bomb
test_bash_execute_blocklist_allows_normal_rm
test_bash_execute_blocklist_allows_rm_rf_subdir

# Concurrency
test_bash_execute_multiple_concurrent_commands
test_bash_execute_tracks_running_processes

# Duration tracking
test_bash_execute_returns_duration_ms
```

File: `tests/conftest.py` (new fixtures)
```
workspace_with_files  — workspace with sample files for bash tests
safe_env              — monkeypatched environment with dummy tokens to verify they're stripped
```

## Implementation Notes

1. **Module location:** `src/chorus/tools/bash.py`.

2. **Core execution function:**
   ```python
   async def bash_execute(
       command: str,
       workspace: Path,
       profile: PermissionProfile,
       timeout: float = 120.0,
       max_output_len: int = 50_000,
       env_overrides: dict[str, str] | None = None,
   ) -> BashResult:
   ```

3. **Environment sanitization — build an explicit allowlist, not a blocklist:**
   ```python
   ALLOWED_ENV_VARS = {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR"}

   def _sanitized_env(workspace: Path) -> dict[str, str]:
       env = {k: v for k, v in os.environ.items() if k in ALLOWED_ENV_VARS}
       env["HOME"] = str(workspace)  # Jail HOME to workspace
       return env
   ```

4. **Timeout implementation:**
   ```python
   process = await asyncio.create_subprocess_shell(
       command,
       cwd=workspace,
       stdout=asyncio.subprocess.PIPE,
       stderr=asyncio.subprocess.PIPE,
       env=sanitized_env,
   )
   try:
       stdout, stderr = await asyncio.wait_for(
           process.communicate(), timeout=timeout
       )
   except asyncio.TimeoutError:
       process.terminate()
       await asyncio.sleep(2)  # Grace period
       if process.returncode is None:
           process.kill()
       return BashResult(timed_out=True, ...)
   ```

5. **Command blocklist** — pattern-based, applied before permission check:
   ```python
   BLOCKED_PATTERNS = [
       re.compile(r"rm\s+-[^\s]*r[^\s]*f[^\s]*\s+/\s*$"),  # rm -rf /
       re.compile(r":\(\)\s*\{.*\}"),                         # fork bomb
       re.compile(r"dd\s+if=/dev/(zero|random)"),             # disk fill
       re.compile(r"mkfs"),                                    # format disk
       re.compile(r">\s*/dev/sd[a-z]"),                        # overwrite disk
   ]
   ```
   These are best-effort. Log blocked attempts to the audit table.

6. **BashResult dataclass:**
   ```python
   @dataclass
   class BashResult:
       command: str
       exit_code: int | None
       stdout: str
       stderr: str
       timed_out: bool
       duration_ms: int
       truncated: bool = False
   ```

7. **Process tracking:** Maintain a per-agent dict of running `asyncio.subprocess.Process` objects. This supports the `/tasks` command and allows cleanup on agent destroy. Use an `asyncio.Lock` for thread-safe access.

8. **Output truncation:** Truncate from the front (keep the tail) — the most recent output is usually more useful. Include a header: `"[Output truncated: showing last {max_output_len} chars of {total} chars]\n"`.

9. **Testing:** Most tests can run real subprocesses (e.g., `echo hello`, `ls`, `exit 1`). For timeout tests, use `sleep 10` with a short timeout. For blocklist tests, verify the command is rejected before any subprocess is created.

## Dependencies

- **002-agent-manager**: Agent workspace directory must exist.
- **003-permission-profiles**: Permission checks for bash commands.
