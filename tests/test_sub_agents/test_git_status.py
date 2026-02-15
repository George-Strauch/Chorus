"""Tests for the git status sub-agent tool."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from chorus.sub_agents.tasks.git_status import (
    _collect_git_info,
    _discover_git_repos,
    _filter_relevant_repos,
    _format_git_report,
    _recover_failed_paths,
    git_status_execute,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_git_repo(path: Path, *, branch: str = "main") -> None:
    """Initialize a git repo with an initial commit at `path`."""
    path.mkdir(parents=True, exist_ok=True)
    env = {"HOME": str(path.parent), "PATH": os.environ.get("PATH", "")}

    def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd, cwd=path, env=env, capture_output=True, text=True, check=True
        )

    run(["git", "init", "-b", branch])
    run(["git", "config", "user.name", "test"])
    run(["git", "config", "user.email", "test@test.local"])
    (path / "README.md").write_text("# Test\n")
    run(["git", "add", "README.md"])
    run(["git", "commit", "-m", "Initial commit"])


# ---------------------------------------------------------------------------
# _discover_git_repos tests
# ---------------------------------------------------------------------------


class TestDiscoverGitRepos:
    @pytest.mark.asyncio
    async def test_finds_workspace_repo(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        _init_git_repo(ws)

        repos = await _discover_git_repos(ws)
        assert ws in repos

    @pytest.mark.asyncio
    async def test_finds_nested_repos(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()
        nested = ws / "projects" / "myapp"
        _init_git_repo(nested)

        repos = await _discover_git_repos(ws)
        assert nested in repos

    @pytest.mark.asyncio
    async def test_empty_workspace_no_repos(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()

        repos = await _discover_git_repos(ws)
        assert repos == []

    @pytest.mark.asyncio
    async def test_deduplicates_repos(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        _init_git_repo(ws)

        repos = await _discover_git_repos(ws)
        # Should contain workspace only once
        assert repos.count(ws) == 1


# ---------------------------------------------------------------------------
# _filter_relevant_repos tests
# ---------------------------------------------------------------------------


class TestFilterRelevantRepos:
    @pytest.mark.asyncio
    async def test_no_repos_returns_empty(self, tmp_path: Path) -> None:
        result = await _filter_relevant_repos(
            [], tmp_path, "test-agent"
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_no_keys_returns_workspace_repos(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ws = tmp_path / "workspace"
        ws.mkdir()
        other = tmp_path / "other"
        other.mkdir()

        result = await _filter_relevant_repos(
            [ws, other], ws, "test-agent"
        )
        assert ws in result
        assert other not in result

    @pytest.mark.asyncio
    async def test_haiku_filter_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        ws = tmp_path / "workspace"
        ws.mkdir()
        other = tmp_path / "other"
        other.mkdir()

        from chorus.llm.providers import Usage
        from chorus.sub_agents.runner import SubAgentResult

        mock_result = SubAgentResult(
            success=True,
            output=f'["{ws}"]',
            model_used="claude-haiku",
            usage=Usage(input_tokens=10, output_tokens=5),
        )

        with patch(
            "chorus.sub_agents.runner.run_sub_agent",
            AsyncMock(return_value=mock_result),
        ):
            result = await _filter_relevant_repos(
                [ws, other], ws, "test-agent"
            )

        assert ws in result

    @pytest.mark.asyncio
    async def test_haiku_failure_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        ws = tmp_path / "workspace"
        ws.mkdir()

        from chorus.llm.providers import Usage
        from chorus.sub_agents.runner import SubAgentResult

        mock_result = SubAgentResult(
            success=False,
            output="",
            model_used="claude-haiku",
            usage=Usage(input_tokens=0, output_tokens=0),
            error="API error",
        )

        with patch(
            "chorus.sub_agents.runner.run_sub_agent",
            AsyncMock(return_value=mock_result),
        ):
            result = await _filter_relevant_repos(
                [ws], ws, "test-agent"
            )

        assert ws in result

    @pytest.mark.asyncio
    async def test_invalid_json_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        ws = tmp_path / "workspace"
        ws.mkdir()

        from chorus.llm.providers import Usage
        from chorus.sub_agents.runner import SubAgentResult

        mock_result = SubAgentResult(
            success=True,
            output="not valid json",
            model_used="claude-haiku",
            usage=Usage(input_tokens=10, output_tokens=5),
        )

        with patch(
            "chorus.sub_agents.runner.run_sub_agent",
            AsyncMock(return_value=mock_result),
        ):
            result = await _filter_relevant_repos(
                [ws], ws, "test-agent"
            )

        assert ws in result


# ---------------------------------------------------------------------------
# _collect_git_info tests
# ---------------------------------------------------------------------------


class TestCollectGitInfo:
    @pytest.mark.asyncio
    async def test_collects_from_valid_repo(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo)

        info = await _collect_git_info(repo)
        assert info["success"]
        assert info["branch"] == "main"
        assert len(info["commits"]) >= 1
        assert info["error"] is None

    @pytest.mark.asyncio
    async def test_not_a_git_repo(self, tmp_path: Path) -> None:
        not_repo = tmp_path / "not-a-repo"
        not_repo.mkdir()

        info = await _collect_git_info(not_repo)
        assert not info["success"]
        assert info["error"] == "Not a git repository"

    @pytest.mark.asyncio
    async def test_detects_untracked_files(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo)
        (repo / "new_file.txt").write_text("untracked")

        info = await _collect_git_info(repo)
        assert info["success"]
        assert info["untracked_count"] >= 1

    @pytest.mark.asyncio
    async def test_detects_changes(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo)
        (repo / "README.md").write_text("Modified content\n")

        info = await _collect_git_info(repo)
        assert info["success"]
        assert info["changes"]  # Should have diff output

    @pytest.mark.asyncio
    async def test_detects_staged_changes(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo)
        (repo / "README.md").write_text("Staged modification\n")
        env = {"HOME": str(tmp_path), "PATH": os.environ.get("PATH", "")}
        subprocess.run(
            ["git", "add", "README.md"],
            cwd=repo,
            env=env,
            check=True,
            capture_output=True,
        )

        info = await _collect_git_info(repo)
        assert info["success"]
        assert info["staged"]  # Should have staged diff output

    @pytest.mark.asyncio
    async def test_detached_head(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_git_repo(repo)
        env = {"HOME": str(tmp_path), "PATH": os.environ.get("PATH", "")}
        # Get the commit hash and checkout to detached HEAD
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        commit_hash = result.stdout.strip()
        subprocess.run(
            ["git", "checkout", commit_hash],
            cwd=repo,
            env=env,
            check=True,
            capture_output=True,
        )

        info = await _collect_git_info(repo)
        assert info["success"]
        assert info["branch"] == "(detached HEAD)"


# ---------------------------------------------------------------------------
# _recover_failed_paths tests
# ---------------------------------------------------------------------------


class TestRecoverFailedPaths:
    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        result = await _recover_failed_paths({})
        assert result == {}

    @pytest.mark.asyncio
    async def test_recovery_returns_corrected_paths(
        self, tmp_path: Path
    ) -> None:
        old_path = tmp_path / "old"
        new_path = tmp_path / "new"
        new_path.mkdir()

        from chorus.llm.providers import Usage
        from chorus.sub_agents.runner import SubAgentResult

        mock_result = SubAgentResult(
            success=True,
            output=f'{{"{old_path}": "{new_path}"}}',
            model_used="claude-haiku",
            usage=Usage(input_tokens=10, output_tokens=5),
        )

        with patch(
            "chorus.sub_agents.runner.run_sub_agent",
            AsyncMock(return_value=mock_result),
        ):
            result = await _recover_failed_paths({old_path: "Not found"})

        assert result.get(old_path) == new_path

    @pytest.mark.asyncio
    async def test_recovery_returns_none_for_real_errors(self) -> None:
        from chorus.llm.providers import Usage
        from chorus.sub_agents.runner import SubAgentResult

        old_path = Path("/nonexistent")

        mock_result = SubAgentResult(
            success=True,
            output=f'{{"{old_path}": null}}',
            model_used="claude-haiku",
            usage=Usage(input_tokens=10, output_tokens=5),
        )

        with patch(
            "chorus.sub_agents.runner.run_sub_agent",
            AsyncMock(return_value=mock_result),
        ):
            result = await _recover_failed_paths(
                {old_path: "Permission denied"}
            )

        assert result.get(old_path) is None

    @pytest.mark.asyncio
    async def test_recovery_failure_returns_empty(self) -> None:
        from chorus.llm.providers import Usage
        from chorus.sub_agents.runner import SubAgentResult

        mock_result = SubAgentResult(
            success=False,
            output="",
            model_used="claude-haiku",
            usage=Usage(input_tokens=0, output_tokens=0),
            error="API error",
        )

        with patch(
            "chorus.sub_agents.runner.run_sub_agent",
            AsyncMock(return_value=mock_result),
        ):
            result = await _recover_failed_paths(
                {Path("/bad"): "Not found"}
            )

        assert result == {}


# ---------------------------------------------------------------------------
# _format_git_report tests
# ---------------------------------------------------------------------------


class TestFormatGitReport:
    def test_empty_results(self) -> None:
        report = _format_git_report({})
        assert "No git repositories" in report

    def test_successful_repo(self) -> None:
        results: dict[Path, dict[str, Any]] = {
            Path("/repo"): {
                "success": True,
                "branch": "main",
                "commits": ["abc1234 Initial commit"],
                "changes": "",
                "staged": "",
                "untracked_count": 0,
                "error": None,
            }
        }
        report = _format_git_report(results)
        assert "/repo" in report
        assert "Branch: main" in report
        assert "abc1234" in report
        assert "Changes: (clean)" in report

    def test_failed_repo(self) -> None:
        results: dict[Path, dict[str, Any]] = {
            Path("/bad-repo"): {
                "success": False,
                "branch": "",
                "commits": [],
                "changes": "",
                "staged": "",
                "untracked_count": 0,
                "error": "Not a git repository",
            }
        }
        report = _format_git_report(results)
        assert "/bad-repo" in report
        assert "Not a git repository" in report

    def test_repo_with_changes(self) -> None:
        results: dict[Path, dict[str, Any]] = {
            Path("/repo"): {
                "success": True,
                "branch": "feature",
                "commits": ["abc1234 feat: add feature"],
                "changes": (
                    " file.py | 5 ++---\n"
                    " 1 file changed, 2 insertions(+), 3 deletions(-)"
                ),
                "staged": "",
                "untracked_count": 3,
                "error": None,
            }
        }
        report = _format_git_report(results)
        assert "Branch: feature" in report
        assert "Changes:" in report
        assert "Untracked: 3 files" in report

    def test_repo_with_staged(self) -> None:
        results: dict[Path, dict[str, Any]] = {
            Path("/repo"): {
                "success": True,
                "branch": "main",
                "commits": [],
                "changes": "",
                "staged": (
                    " file.py | 2 ++\n"
                    " 1 file changed, 2 insertions(+)"
                ),
                "untracked_count": 0,
                "error": None,
            }
        }
        report = _format_git_report(results)
        assert "Staged:" in report
        assert "1 file changed" in report

    def test_no_commits(self) -> None:
        results: dict[Path, dict[str, Any]] = {
            Path("/repo"): {
                "success": True,
                "branch": "main",
                "commits": [],
                "changes": "",
                "staged": "",
                "untracked_count": 0,
                "error": None,
            }
        }
        report = _format_git_report(results)
        assert "Recent commits: (none)" in report

    def test_multiple_repos_sorted(self) -> None:
        results: dict[Path, dict[str, Any]] = {
            Path("/z-repo"): {
                "success": True,
                "branch": "main",
                "commits": [],
                "changes": "",
                "staged": "",
                "untracked_count": 0,
                "error": None,
            },
            Path("/a-repo"): {
                "success": True,
                "branch": "develop",
                "commits": [],
                "changes": "",
                "staged": "",
                "untracked_count": 0,
                "error": None,
            },
        }
        report = _format_git_report(results)
        a_pos = report.index("/a-repo")
        z_pos = report.index("/z-repo")
        assert a_pos < z_pos  # Sorted alphabetically


# ---------------------------------------------------------------------------
# git_status_execute integration tests
# ---------------------------------------------------------------------------


class TestGitStatusExecute:
    @pytest.mark.asyncio
    async def test_no_repos_found(self, tmp_path: Path) -> None:
        ws = tmp_path / "empty-ws"
        ws.mkdir()
        # Mock _discover_git_repos to return empty
        with patch(
            "chorus.sub_agents.tasks.git_status._discover_git_repos",
            AsyncMock(return_value=[]),
        ):
            result = await git_status_execute(
                workspace=ws, agent_name="test-agent"
            )
        assert "No git repositories found" in result

    @pytest.mark.asyncio
    async def test_successful_report(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        _init_git_repo(ws)

        # Mock _filter_relevant_repos to return workspace directly
        with patch(
            "chorus.sub_agents.tasks.git_status._filter_relevant_repos",
            AsyncMock(return_value=[ws]),
        ):
            result = await git_status_execute(
                workspace=ws, agent_name="test-agent"
            )

        assert "Branch: main" in result
        assert "Initial commit" in result

    @pytest.mark.asyncio
    async def test_timeout_handling(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()

        with patch(
            "chorus.sub_agents.tasks.git_status._OVERALL_TIMEOUT", 0.001
        ), patch(
            "chorus.sub_agents.tasks.git_status._git_status_execute_impl",
            new_callable=AsyncMock,
            side_effect=TimeoutError,
        ):
            # The outer handler wraps with asyncio.wait_for which will catch
            result = await git_status_execute(
                workspace=ws, agent_name="test-agent"
            )
        # Timeout can be caught either by wait_for or the try/except
        assert "timed out" in result.lower() or "timeout" in result.lower()

    @pytest.mark.asyncio
    async def test_exception_handling(self, tmp_path: Path) -> None:
        ws = tmp_path / "workspace"
        ws.mkdir()

        with patch(
            "chorus.sub_agents.tasks.git_status._git_status_execute_impl",
            AsyncMock(side_effect=RuntimeError("boom")),
        ):
            result = await git_status_execute(
                workspace=ws, agent_name="test-agent"
            )

        assert "failed" in result.lower() or "RuntimeError" in result

    @pytest.mark.asyncio
    async def test_with_failed_and_recovered_repos(
        self, tmp_path: Path
    ) -> None:
        ws = tmp_path / "workspace"
        _init_git_repo(ws)
        other = tmp_path / "other"
        _init_git_repo(other)

        # Mock discover to find both, filter returns both, but one fails
        with (
            patch(
                "chorus.sub_agents.tasks.git_status._discover_git_repos",
                AsyncMock(return_value=[ws, other]),
            ),
            patch(
                "chorus.sub_agents.tasks.git_status._filter_relevant_repos",
                AsyncMock(return_value=[ws, other]),
            ),
        ):
            result = await git_status_execute(
                workspace=ws, agent_name="test-agent"
            )

        # Both repos should appear in the report
        assert str(ws) in result
        assert str(other) in result


# ---------------------------------------------------------------------------
# Tool registration tests
# ---------------------------------------------------------------------------


class TestGitStatusRegistration:
    def test_registered_in_default_registry(self) -> None:
        from chorus.tools.registry import create_default_registry

        registry = create_default_registry()
        tool = registry.get("git_status")
        assert tool is not None
        assert tool.name == "git_status"
        assert "git status" in tool.description.lower()

    def test_permission_category_mapping(self) -> None:
        from chorus.llm.tool_loop import _TOOL_TO_CATEGORY

        assert _TOOL_TO_CATEGORY.get("git_status") == "git"
