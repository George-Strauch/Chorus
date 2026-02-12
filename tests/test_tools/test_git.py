"""Tests for chorus.tools.git — git operations in agent workspace."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from chorus.permissions.engine import PermissionProfile, get_preset
from chorus.tools.bash import CommandDeniedError, CommandNeedsApprovalError
from chorus.tools.git import (
    GitError,
    UnsupportedForgeError,
    git_branch,
    git_checkout,
    git_commit,
    git_diff,
    git_init,
    git_log,
    git_merge_request,
    git_push,
)

OPEN = get_preset("open")
STANDARD = get_preset("standard")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_git(workspace: Path, *args: str) -> str:
    """Synchronous helper to run git commands in a workspace."""
    result = subprocess.run(
        ["git", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# TestGitInit
# ---------------------------------------------------------------------------


class TestGitInit:
    @pytest.mark.asyncio
    async def test_creates_repo(self, tmp_path: Path) -> None:
        ws = tmp_path / "new-workspace"
        ws.mkdir()
        result = await git_init(ws, "my-agent", OPEN)
        assert result.success
        assert (ws / ".git").is_dir()

    @pytest.mark.asyncio
    async def test_sets_user_config(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        await git_init(ws, "my-agent", OPEN)
        name = _run_git(ws, "config", "user.name")
        email = _run_git(ws, "config", "user.email")
        assert name == "my-agent"
        assert email == "my-agent@chorus.local"

    @pytest.mark.asyncio
    async def test_idempotent_reinit(self, git_workspace: Path) -> None:
        # git_workspace already has a repo — reinit should succeed
        result = await git_init(git_workspace, "test-agent", OPEN)
        assert result.success
        # Existing commits should survive
        log = _run_git(git_workspace, "log", "--oneline")
        assert "Initial commit" in log


# ---------------------------------------------------------------------------
# TestGitCommit
# ---------------------------------------------------------------------------


class TestGitCommit:
    @pytest.mark.asyncio
    async def test_stages_and_commits(self, git_workspace: Path) -> None:
        (git_workspace / "new_file.txt").write_text("hello\n")
        result = await git_commit(git_workspace, "Add new file", OPEN)
        assert result.success
        assert result.operation == "commit"
        log = _run_git(git_workspace, "log", "--oneline")
        assert "Add new file" in log

    @pytest.mark.asyncio
    async def test_specific_files(self, git_workspace: Path) -> None:
        (git_workspace / "a.txt").write_text("aaa\n")
        (git_workspace / "b.txt").write_text("bbb\n")
        result = await git_commit(
            git_workspace, "Only add a.txt", OPEN, files=["a.txt"]
        )
        assert result.success
        # b.txt should NOT be committed
        status = _run_git(git_workspace, "status", "--porcelain")
        assert "b.txt" in status
        assert "a.txt" not in status

    @pytest.mark.asyncio
    async def test_returns_commit_hash(self, git_workspace: Path) -> None:
        (git_workspace / "for_hash.txt").write_text("data\n")
        result = await git_commit(git_workspace, "Commit for hash", OPEN)
        assert result.commit_hash is not None
        assert len(result.commit_hash) >= 7
        # Verify the hash exists in the repo
        _run_git(git_workspace, "rev-parse", result.commit_hash)

    @pytest.mark.asyncio
    async def test_empty_tree_fails(self, git_workspace: Path) -> None:
        # Nothing to commit
        result = await git_commit(git_workspace, "Empty commit", OPEN)
        assert not result.success

    @pytest.mark.asyncio
    async def test_permission_check(self, git_workspace: Path) -> None:
        deny_profile = PermissionProfile(allow=[], ask=[])
        (git_workspace / "file.txt").write_text("test\n")
        with pytest.raises(CommandDeniedError):
            await git_commit(git_workspace, "msg", deny_profile)


# ---------------------------------------------------------------------------
# TestGitPush
# ---------------------------------------------------------------------------


class TestGitPush:
    @pytest.mark.asyncio
    async def test_action_string_format(
        self, git_workspace_with_remote: tuple[Path, Path]
    ) -> None:
        ws, _ = git_workspace_with_remote
        (ws / "push_test.txt").write_text("push me\n")
        _run_git(ws, "add", "push_test.txt")
        _run_git(ws, "commit", "-m", "push test")
        result = await git_push(ws, "origin", "main", OPEN)
        assert result.success

    @pytest.mark.asyncio
    async def test_asks_in_standard_profile(
        self, git_workspace_with_remote: tuple[Path, Path]
    ) -> None:
        ws, _ = git_workspace_with_remote
        with pytest.raises(CommandNeedsApprovalError):
            await git_push(ws, "origin", "main", STANDARD)

    @pytest.mark.asyncio
    async def test_allowed_in_open_profile(
        self, git_workspace_with_remote: tuple[Path, Path]
    ) -> None:
        ws, _ = git_workspace_with_remote
        # No new commits, but push should still succeed (nothing to push is OK)
        result = await git_push(ws, "origin", "main", OPEN)
        assert result.success


# ---------------------------------------------------------------------------
# TestGitBranch
# ---------------------------------------------------------------------------


class TestGitBranch:
    @pytest.mark.asyncio
    async def test_create(self, git_workspace: Path) -> None:
        result = await git_branch(git_workspace, OPEN, branch_name="feature-x")
        assert result.success
        branches = _run_git(git_workspace, "branch", "--list")
        assert "feature-x" in branches

    @pytest.mark.asyncio
    async def test_list(self, git_workspace: Path) -> None:
        _run_git(git_workspace, "branch", "branch-a")
        _run_git(git_workspace, "branch", "branch-b")
        result = await git_branch(git_workspace, OPEN)
        assert result.success
        assert "branch-a" in result.stdout
        assert "branch-b" in result.stdout

    @pytest.mark.asyncio
    async def test_delete(self, git_workspace: Path) -> None:
        _run_git(git_workspace, "branch", "to-delete")
        result = await git_branch(
            git_workspace, OPEN, branch_name="to-delete", delete=True
        )
        assert result.success
        branches = _run_git(git_workspace, "branch", "--list")
        assert "to-delete" not in branches

    @pytest.mark.asyncio
    async def test_already_exists_error(self, git_workspace: Path) -> None:
        _run_git(git_workspace, "branch", "existing")
        result = await git_branch(git_workspace, OPEN, branch_name="existing")
        assert not result.success
        assert result.stderr  # git reports an error


# ---------------------------------------------------------------------------
# TestGitCheckout
# ---------------------------------------------------------------------------


class TestGitCheckout:
    @pytest.mark.asyncio
    async def test_existing_branch(self, git_workspace: Path) -> None:
        _run_git(git_workspace, "branch", "dev")
        result = await git_checkout(git_workspace, "dev", OPEN)
        assert result.success
        current = _run_git(git_workspace, "branch", "--show-current")
        assert current == "dev"

    @pytest.mark.asyncio
    async def test_create_with_b_flag(self, git_workspace: Path) -> None:
        result = await git_checkout(
            git_workspace, "new-branch", OPEN, create=True
        )
        assert result.success
        current = _run_git(git_workspace, "branch", "--show-current")
        assert current == "new-branch"

    @pytest.mark.asyncio
    async def test_nonexistent_ref_error(self, git_workspace: Path) -> None:
        result = await git_checkout(git_workspace, "no-such-branch", OPEN)
        assert not result.success


# ---------------------------------------------------------------------------
# TestGitDiff
# ---------------------------------------------------------------------------


class TestGitDiff:
    @pytest.mark.asyncio
    async def test_working_tree_vs_head(self, git_workspace: Path) -> None:
        (git_workspace / "README.md").write_text("# Updated\n")
        result = await git_diff(git_workspace, OPEN)
        assert result.success
        assert "Updated" in result.stdout

    @pytest.mark.asyncio
    async def test_between_refs(self, git_workspace: Path) -> None:
        (git_workspace / "new.txt").write_text("content\n")
        _run_git(git_workspace, "add", "new.txt")
        _run_git(git_workspace, "commit", "-m", "second commit")
        # Diff between initial and current HEAD
        result = await git_diff(git_workspace, OPEN, ref1="HEAD~1", ref2="HEAD")
        assert result.success
        assert "new.txt" in result.stdout

    @pytest.mark.asyncio
    async def test_no_changes(self, git_workspace: Path) -> None:
        result = await git_diff(git_workspace, OPEN)
        assert result.success
        assert result.stdout.strip() == ""

    @pytest.mark.asyncio
    async def test_returns_structured_output(self, git_workspace: Path) -> None:
        (git_workspace / "README.md").write_text("# Changed\n")
        result = await git_diff(git_workspace, OPEN)
        assert result.operation == "diff"
        d = result.to_dict()
        assert "operation" in d
        assert "stdout" in d


# ---------------------------------------------------------------------------
# TestGitLog
# ---------------------------------------------------------------------------


class TestGitLog:
    @pytest.mark.asyncio
    async def test_default_count(self, git_workspace: Path) -> None:
        result = await git_log(git_workspace, OPEN)
        assert result.success
        assert "Initial commit" in result.stdout

    @pytest.mark.asyncio
    async def test_custom_count(self, git_workspace: Path) -> None:
        # Add a few commits
        for i in range(3):
            (git_workspace / f"file{i}.txt").write_text(f"content {i}\n")
            _run_git(git_workspace, "add", f"file{i}.txt")
            _run_git(git_workspace, "commit", "-m", f"Commit {i}")
        result = await git_log(git_workspace, OPEN, count=2)
        assert result.success
        # Should have at most 2 commit entries in the log
        assert "Commit 2" in result.stdout
        assert "Commit 1" in result.stdout

    @pytest.mark.asyncio
    async def test_oneline_format(self, git_workspace: Path) -> None:
        result = await git_log(git_workspace, OPEN, oneline=True)
        assert result.success
        lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
        # Oneline: each commit is exactly one line
        assert len(lines) >= 1
        # Each line should have a short hash + message
        assert "Initial commit" in lines[0]

    @pytest.mark.asyncio
    async def test_empty_repo(self, tmp_path: Path) -> None:
        ws = tmp_path / "empty"
        ws.mkdir()
        await git_init(ws, "agent", OPEN)
        result = await git_log(ws, OPEN)
        # Empty repo has no commits — git log returns non-zero
        assert not result.success


# ---------------------------------------------------------------------------
# TestMergeRequest
# ---------------------------------------------------------------------------


class TestMergeRequest:
    @pytest.mark.asyncio
    async def test_github_detection(
        self, git_workspace_with_remote: tuple[Path, Path]
    ) -> None:
        ws, _ = git_workspace_with_remote
        # Override remote to a github URL
        _run_git(ws, "remote", "set-url", "origin", "git@github.com:user/repo.git")
        with patch("chorus.tools.git.bash_execute", new_callable=AsyncMock) as mock_bash:
            from chorus.tools.bash import BashResult

            mock_bash.return_value = BashResult(
                command="gh pr create ...",
                exit_code=0,
                stdout="https://github.com/user/repo/pull/1",
                stderr="",
                timed_out=False,
                duration_ms=100,
            )
            result = await git_merge_request(
                ws, "My PR", "Description", "feature", "main", OPEN
            )
            assert result.success
            # Verify gh was called
            call_cmd = mock_bash.call_args[0][0]
            assert "gh pr create" in call_cmd

    @pytest.mark.asyncio
    async def test_gitlab_detection(
        self, git_workspace_with_remote: tuple[Path, Path]
    ) -> None:
        ws, _ = git_workspace_with_remote
        _run_git(
            ws, "remote", "set-url", "origin", "git@gitlab.com:user/repo.git"
        )
        with patch("chorus.tools.git.bash_execute", new_callable=AsyncMock) as mock_bash:
            from chorus.tools.bash import BashResult

            mock_bash.return_value = BashResult(
                command="glab mr create ...",
                exit_code=0,
                stdout="https://gitlab.com/user/repo/-/merge_requests/1",
                stderr="",
                timed_out=False,
                duration_ms=100,
            )
            result = await git_merge_request(
                ws, "My MR", "Desc", "feature", "main", OPEN
            )
            assert result.success
            call_cmd = mock_bash.call_args[0][0]
            assert "glab mr create" in call_cmd

    @pytest.mark.asyncio
    async def test_no_remote_error(self, git_workspace: Path) -> None:
        # git_workspace has no remote
        with pytest.raises(GitError):
            await git_merge_request(
                git_workspace, "PR", "Desc", "feature", "main", OPEN
            )

    @pytest.mark.asyncio
    async def test_permission_check(
        self, git_workspace_with_remote: tuple[Path, Path]
    ) -> None:
        ws, _ = git_workspace_with_remote
        _run_git(ws, "remote", "set-url", "origin", "git@github.com:u/r.git")
        with pytest.raises(CommandNeedsApprovalError):
            await git_merge_request(
                ws, "PR", "Desc", "feature", "main", STANDARD
            )

    @pytest.mark.asyncio
    async def test_unsupported_forge(
        self, git_workspace_with_remote: tuple[Path, Path]
    ) -> None:
        ws, _ = git_workspace_with_remote
        _run_git(
            ws,
            "remote",
            "set-url",
            "origin",
            "git@bitbucket.org:user/repo.git",
        )
        with pytest.raises(UnsupportedForgeError):
            await git_merge_request(
                ws, "PR", "Desc", "feature", "main", OPEN
            )


# ---------------------------------------------------------------------------
# TestGeneral
# ---------------------------------------------------------------------------


class TestGeneral:
    @pytest.mark.asyncio
    async def test_runs_in_workspace_dir(self, git_workspace: Path) -> None:
        result = await git_log(git_workspace, OPEN)
        assert result.success
        assert "Initial commit" in result.stdout

    @pytest.mark.asyncio
    async def test_stderr_on_failure(self, git_workspace: Path) -> None:
        result = await git_checkout(git_workspace, "nonexistent-ref", OPEN)
        assert not result.success
        assert result.stderr  # git writes errors to stderr

    @pytest.mark.asyncio
    async def test_to_dict(self, git_workspace: Path) -> None:
        result = await git_log(git_workspace, OPEN)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "operation" in d
        assert "success" in d
        assert "stdout" in d
        assert "stderr" in d

    @pytest.mark.asyncio
    async def test_deny_raises(self, git_workspace: Path) -> None:
        deny = PermissionProfile(allow=[], ask=[])
        with pytest.raises(CommandDeniedError):
            await git_log(git_workspace, deny)

    @pytest.mark.asyncio
    async def test_ask_raises(self, git_workspace: Path) -> None:
        ask_only = PermissionProfile(allow=[], ask=[r"tool:git:.*"])
        with pytest.raises(CommandNeedsApprovalError):
            await git_log(git_workspace, ask_only)
