"""Tests for chorus.tools.bash — sandboxed command execution."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import pytest

from chorus.permissions.engine import PermissionProfile
from chorus.tools.bash import (
    BashResult,
    CommandBlockedError,
    CommandDeniedError,
    CommandNeedsApprovalError,
    _sanitized_env,
    _targets_scope_path,
    bash_execute,
    get_process_tracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOW_ALL = PermissionProfile(allow=[".*"], ask=[])
DENY_ALL = PermissionProfile(allow=[], ask=[])
ASK_ALL = PermissionProfile(allow=[], ask=[".*"])


# ---------------------------------------------------------------------------
# TestBasicExecution
# ---------------------------------------------------------------------------


class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_bash_execute_simple_command(self, workspace_dir: Path) -> None:
        result = await bash_execute("echo hello", workspace_dir, ALLOW_ALL)
        assert isinstance(result, BashResult)
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_bash_execute_captures_stdout(self, workspace_dir: Path) -> None:
        result = await bash_execute("echo hello", workspace_dir, ALLOW_ALL)
        assert result.stdout.strip() == "hello"

    @pytest.mark.asyncio
    async def test_bash_execute_captures_stderr(self, workspace_dir: Path) -> None:
        result = await bash_execute("echo error >&2", workspace_dir, ALLOW_ALL)
        assert "error" in result.stderr

    @pytest.mark.asyncio
    async def test_bash_execute_returns_exit_code(self, workspace_dir: Path) -> None:
        result = await bash_execute("true", workspace_dir, ALLOW_ALL)
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_bash_execute_nonzero_exit_code(self, workspace_dir: Path) -> None:
        result = await bash_execute("exit 42", workspace_dir, ALLOW_ALL)
        assert result.exit_code == 42

    @pytest.mark.asyncio
    async def test_bash_execute_cwd_is_workspace(self, workspace_dir: Path) -> None:
        result = await bash_execute("pwd", workspace_dir, ALLOW_ALL)
        assert result.stdout.strip() == str(workspace_dir.resolve())


# ---------------------------------------------------------------------------
# TestPermissionIntegration
# ---------------------------------------------------------------------------


class TestPermissionIntegration:
    @pytest.mark.asyncio
    async def test_bash_execute_checks_permission_before_running(self, workspace_dir: Path) -> None:
        with pytest.raises(CommandDeniedError):
            await bash_execute("touch should-not-exist", workspace_dir, DENY_ALL)
        assert not (workspace_dir / "should-not-exist").exists()

    @pytest.mark.asyncio
    async def test_bash_execute_denied_command_not_executed(self, workspace_dir: Path) -> None:
        with pytest.raises(CommandDeniedError):
            await bash_execute("echo denied", workspace_dir, DENY_ALL)

    @pytest.mark.asyncio
    async def test_bash_execute_ask_result_returned_to_caller(self, workspace_dir: Path) -> None:
        with pytest.raises(CommandNeedsApprovalError) as exc_info:
            await bash_execute("echo ask-me", workspace_dir, ASK_ALL)
        assert exc_info.value.command == "echo ask-me"


# ---------------------------------------------------------------------------
# TestTimeout
# ---------------------------------------------------------------------------


class TestTimeout:
    @pytest.mark.asyncio
    async def test_bash_execute_timeout_kills_process(self, workspace_dir: Path) -> None:
        result = await bash_execute("sleep 30", workspace_dir, ALLOW_ALL, timeout=0.5)
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_bash_execute_timeout_returns_timed_out_flag(self, workspace_dir: Path) -> None:
        result = await bash_execute("sleep 30", workspace_dir, ALLOW_ALL, timeout=0.5)
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_bash_execute_long_running_completes_within_timeout(
        self, workspace_dir: Path
    ) -> None:
        result = await bash_execute("sleep 0.1", workspace_dir, ALLOW_ALL, timeout=5.0)
        assert result.timed_out is False
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_bash_execute_sigterm_then_sigkill(self, workspace_dir: Path) -> None:
        # trap '' TERM ignores SIGTERM, forcing escalation to SIGKILL
        result = await bash_execute(
            "trap '' TERM; sleep 30", workspace_dir, ALLOW_ALL, timeout=0.5, sigterm_grace=1.0
        )
        assert result.timed_out is True


# ---------------------------------------------------------------------------
# TestOutputHandling
# ---------------------------------------------------------------------------


class TestOutputHandling:
    @pytest.mark.asyncio
    async def test_bash_execute_truncates_long_stdout(self, workspace_dir: Path) -> None:
        result = await bash_execute(
            "python3 -c \"print('x' * 500)\"",
            workspace_dir,
            ALLOW_ALL,
            max_output_len=100,
        )
        assert result.truncated is True
        assert len(result.stdout) <= 200  # header + 100 chars
        assert "[Output truncated" in result.stdout

    @pytest.mark.asyncio
    async def test_bash_execute_truncates_long_stderr(self, workspace_dir: Path) -> None:
        result = await bash_execute(
            "python3 -c \"import sys; sys.stderr.write('e' * 500)\"",
            workspace_dir,
            ALLOW_ALL,
            max_output_len=100,
        )
        assert result.truncated is True
        assert "[Output truncated" in result.stderr

    @pytest.mark.asyncio
    async def test_bash_execute_empty_output(self, workspace_dir: Path) -> None:
        result = await bash_execute("true", workspace_dir, ALLOW_ALL)
        assert result.stdout == ""
        assert result.stderr == ""

    @pytest.mark.asyncio
    async def test_bash_execute_binary_output_handled(self, workspace_dir: Path) -> None:
        result = await bash_execute("printf '\\x00\\x01\\x02'", workspace_dir, ALLOW_ALL)
        assert isinstance(result.stdout, str)


# ---------------------------------------------------------------------------
# TestEnvironment
# ---------------------------------------------------------------------------


class TestEnvironment:
    @pytest.mark.asyncio
    async def test_bash_execute_env_does_not_contain_discord_token(
        self, workspace_dir: Path, safe_env: None
    ) -> None:
        result = await bash_execute("env", workspace_dir, ALLOW_ALL)
        assert "DISCORD_TOKEN" not in result.stdout

    @pytest.mark.asyncio
    async def test_bash_execute_env_does_not_contain_api_keys(
        self, workspace_dir: Path, safe_env: None
    ) -> None:
        result = await bash_execute("env", workspace_dir, ALLOW_ALL)
        assert "ANTHROPIC_API_KEY" not in result.stdout
        assert "OPENAI_API_KEY" not in result.stdout

    @pytest.mark.asyncio
    async def test_bash_execute_env_contains_path(
        self, workspace_dir: Path, safe_env: None
    ) -> None:
        result = await bash_execute("echo $PATH", workspace_dir, ALLOW_ALL)
        assert result.stdout.strip() != ""

    @pytest.mark.asyncio
    async def test_bash_execute_env_contains_home(
        self, workspace_dir: Path, safe_env: None
    ) -> None:
        result = await bash_execute("echo $HOME", workspace_dir, ALLOW_ALL)
        assert result.stdout.strip() == str(workspace_dir)

    @pytest.mark.asyncio
    async def test_bash_execute_scope_path_available_when_set(
        self, workspace_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SCOPE_PATH", "/mnt/host")
        result = await bash_execute("echo $SCOPE_PATH", workspace_dir, ALLOW_ALL)
        assert result.stdout.strip() == "/mnt/host"

    @pytest.mark.asyncio
    async def test_bash_execute_scope_path_absent_when_not_set(
        self, workspace_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SCOPE_PATH", raising=False)
        result = await bash_execute("echo $SCOPE_PATH", workspace_dir, ALLOW_ALL)
        assert result.stdout.strip() == ""

    @pytest.mark.asyncio
    async def test_bash_execute_env_sets_pythonunbuffered(
        self, workspace_dir: Path, safe_env: None
    ) -> None:
        """PYTHONUNBUFFERED=1 is set so Python subprocesses flush immediately."""
        result = await bash_execute("echo $PYTHONUNBUFFERED", workspace_dir, ALLOW_ALL)
        assert result.stdout.strip() == "1"


# ---------------------------------------------------------------------------
# TestBlocklist
# ---------------------------------------------------------------------------


class TestBlocklist:
    @pytest.mark.asyncio
    async def test_bash_execute_blocklist_rejects_rm_rf_root(self, workspace_dir: Path) -> None:
        with pytest.raises(CommandBlockedError):
            await bash_execute("rm -rf /", workspace_dir, ALLOW_ALL)

    @pytest.mark.asyncio
    async def test_bash_execute_blocklist_rejects_fork_bomb(self, workspace_dir: Path) -> None:
        with pytest.raises(CommandBlockedError):
            await bash_execute(":(){ :|:& };:", workspace_dir, ALLOW_ALL)

    @pytest.mark.asyncio
    async def test_bash_execute_blocklist_allows_normal_rm(self, workspace_dir: Path) -> None:
        (workspace_dir / "deleteme.txt").write_text("bye")
        result = await bash_execute("rm deleteme.txt", workspace_dir, ALLOW_ALL)
        assert result.exit_code == 0
        assert not (workspace_dir / "deleteme.txt").exists()

    @pytest.mark.asyncio
    async def test_bash_execute_blocklist_allows_rm_rf_subdir(self, workspace_dir: Path) -> None:
        subdir = workspace_dir / "subdir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("data")
        result = await bash_execute("rm -rf subdir", workspace_dir, ALLOW_ALL)
        assert result.exit_code == 0
        assert not subdir.exists()


# ---------------------------------------------------------------------------
# TestConcurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_bash_execute_multiple_concurrent_commands(self, workspace_dir: Path) -> None:
        results = await asyncio.gather(
            bash_execute("echo one", workspace_dir, ALLOW_ALL),
            bash_execute("echo two", workspace_dir, ALLOW_ALL),
            bash_execute("echo three", workspace_dir, ALLOW_ALL),
        )
        assert all(r.exit_code == 0 for r in results)
        outputs = {r.stdout.strip() for r in results}
        assert outputs == {"one", "two", "three"}

    @pytest.mark.asyncio
    async def test_bash_execute_tracks_running_processes(self, workspace_dir: Path) -> None:
        tracker = get_process_tracker()
        agent = "track-test-agent"

        # Before: no running processes
        assert len(tracker.get_running(agent)) == 0

        # Start a long-running command; check tracker mid-execution
        async def run_and_check() -> BashResult:
            return await bash_execute(
                "sleep 5",
                workspace_dir,
                ALLOW_ALL,
                timeout=5.0,
                agent_name=agent,
            )

        task = asyncio.create_task(run_and_check())
        await asyncio.sleep(0.2)  # Let the process start
        running = tracker.get_running(agent)
        assert len(running) >= 1

        # Clean up
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        # Give tracker a moment to unregister
        await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# TestDuration
# ---------------------------------------------------------------------------


class TestDuration:
    @pytest.mark.asyncio
    async def test_bash_execute_returns_duration_ms(self, workspace_dir: Path) -> None:
        result = await bash_execute("sleep 0.1", workspace_dir, ALLOW_ALL)
        assert result.duration_ms >= 50
        assert result.duration_ms < 5000


# ---------------------------------------------------------------------------
# TestTargetsScopePath
# ---------------------------------------------------------------------------


class TestTargetsScopePath:
    def test_returns_false_when_scope_path_is_none(self, workspace_dir: Path) -> None:
        assert _targets_scope_path("cd /mnt/host", workspace_dir, None) is False

    def test_detects_workspace_under_scope_path(self, tmp_path: Path) -> None:
        scope = tmp_path / "host"
        scope.mkdir()
        workspace = scope / "projects" / "my-repo"
        workspace.mkdir(parents=True)
        assert _targets_scope_path("echo hello", workspace, scope) is True

    def test_detects_scope_path_literal_in_command(self, workspace_dir: Path) -> None:
        scope = Path("/mnt/host")
        assert _targets_scope_path("cd /mnt/host/repo && git push", workspace_dir, scope) is True

    def test_detects_scope_path_env_var_dollar(self, workspace_dir: Path) -> None:
        scope = Path("/mnt/host")
        assert _targets_scope_path("cd $SCOPE_PATH/repo", workspace_dir, scope) is True

    def test_detects_scope_path_env_var_braces(self, workspace_dir: Path) -> None:
        scope = Path("/mnt/host")
        assert _targets_scope_path("cd ${SCOPE_PATH}/repo", workspace_dir, scope) is True

    def test_returns_false_for_unrelated_command(self, workspace_dir: Path) -> None:
        scope = Path("/mnt/host")
        assert _targets_scope_path("echo hello", workspace_dir, scope) is False


# ---------------------------------------------------------------------------
# TestSanitizedEnvScopeHome
# ---------------------------------------------------------------------------


class TestSanitizedEnvScopeHome:
    def test_scope_home_overrides_home_in_sandboxed_mode(self, workspace_dir: Path) -> None:
        scope = Path("/mnt/host")
        env = _sanitized_env(workspace_dir, scope_home=scope)
        assert env["HOME"] == "/mnt/host"

    def test_scope_home_overrides_home_in_host_exec_mode(self, workspace_dir: Path) -> None:
        scope = Path("/mnt/host")
        env = _sanitized_env(workspace_dir, host_execution=True, scope_home=scope)
        assert env["HOME"] == "/mnt/host"

    def test_no_scope_home_sandboxed_uses_workspace(self, workspace_dir: Path) -> None:
        env = _sanitized_env(workspace_dir)
        assert env["HOME"] == str(workspace_dir)

    def test_no_scope_home_host_exec_preserves_real_home(self, workspace_dir: Path) -> None:
        import os
        env = _sanitized_env(workspace_dir, host_execution=True)
        assert env["HOME"] == os.environ.get("HOME", "")


# ---------------------------------------------------------------------------
# TestBashExecuteScopePath
# ---------------------------------------------------------------------------


class TestBashExecuteScopePath:
    @pytest.mark.asyncio
    async def test_scope_path_auto_enables_full_env_for_matching_command(
        self, tmp_path: Path
    ) -> None:
        scope = tmp_path / "host"
        scope.mkdir()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        # Command references scope_path — should get full env + HOME=scope
        result = await bash_execute(
            f"echo $HOME",
            workspace,
            ALLOW_ALL,
            scope_path=scope,
        )
        # HOME should NOT be scope because command doesn't reference scope_path
        assert result.stdout.strip() == str(workspace)

    @pytest.mark.asyncio
    async def test_scope_path_sets_home_when_command_references_scope(
        self, tmp_path: Path
    ) -> None:
        scope = tmp_path / "host"
        scope.mkdir()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        sp = str(scope)
        result = await bash_execute(
            f"cd {sp} && echo $HOME",
            workspace,
            ALLOW_ALL,
            scope_path=scope,
        )
        assert result.stdout.strip() == sp

    @pytest.mark.asyncio
    async def test_scope_path_sets_home_when_workspace_under_scope(
        self, tmp_path: Path
    ) -> None:
        scope = tmp_path / "host"
        scope.mkdir()
        workspace = scope / "projects" / "repo"
        workspace.mkdir(parents=True)
        result = await bash_execute(
            "echo $HOME",
            workspace,
            ALLOW_ALL,
            scope_path=scope,
        )
        assert result.stdout.strip() == str(scope)

    @pytest.mark.asyncio
    async def test_scope_path_none_keeps_default_behavior(
        self, workspace_dir: Path
    ) -> None:
        result = await bash_execute(
            "echo $HOME",
            workspace_dir,
            ALLOW_ALL,
            scope_path=None,
        )
        assert result.stdout.strip() == str(workspace_dir)
