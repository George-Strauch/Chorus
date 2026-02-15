"""Advanced git status tool — discovers and reports on relevant repositories.

This tool uses a multi-step approach:
1. Path Discovery (Haiku call) - Find relevant git repos based on agent context
2. Static Script Execution - Collect git status info from each repo
3. Error Recovery (Haiku call, conditional) - Suggest corrected paths for failures
4. Format Output - Combine results into a formatted report
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("chorus.sub_agents.tasks.git_status")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OVERALL_TIMEOUT = 30.0
_DISCOVER_TIMEOUT = 10.0
_ERROR_RECOVERY_TIMEOUT = 10.0
_GIT_COMMAND_TIMEOUT = 5.0

_PATH_DISCOVERY_PROMPT = """Given the agent's workspace and these discovered git repositories,
determine which ones are relevant to the agent's context.

Return ONLY a JSON array of absolute paths (as strings) to the relevant repositories.
Example: ["/mnt/host/PycharmProjects/Chorus", "/mnt/host/Projects/myapp"]

If no repositories are relevant, return an empty array: []
"""

_ERROR_RECOVERY_PROMPT = """These git repository paths failed with errors. For each failed path,
analyze the error message and either:
- Suggest a corrected path if the repo was moved/renamed
- Return null if it's a real error (not a git repo, permission denied, etc.)

Return ONLY valid JSON in this format:
{
  "/old/path": "/corrected/path",
  "/another/path": null,
  "/third/path": "/another/corrected/path"
}
"""

# ---------------------------------------------------------------------------
# Helper: Run shell command
# ---------------------------------------------------------------------------


async def _run_command(
    command: str,
    cwd: Path,
    timeout: float = _GIT_COMMAND_TIMEOUT,
) -> tuple[int, str, str]:
    """Run a shell command and return (exit_code, stdout, stderr).

    Returns (exit_code, stdout, stderr).
    """
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )

        stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
        exit_code = process.returncode or 0

        return exit_code, stdout, stderr

    except TimeoutError:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as exc:
        return -1, "", f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Step 1: Path Discovery
# ---------------------------------------------------------------------------


async def _discover_git_repos(workspace: Path) -> list[Path]:
    """Discover git repositories by scanning workspace and common mount points.

    Returns a list of absolute paths to .git directories (parent dirs of .git).
    """
    discovered: list[Path] = []

    # Search patterns:
    # 1. Workspace directory
    # 2. Common mount points like /mnt/host
    search_roots = [workspace]

    # Add /mnt/host if it exists (common in Docker environments)
    mnt_host = Path("/mnt/host")
    if mnt_host.exists() and mnt_host.is_dir():
        search_roots.append(mnt_host)

    for root in search_roots:
        # Use find to locate .git directories
        # Limit depth to avoid scanning too deep
        cmd = f"find {root} -maxdepth 4 -type d -name .git 2>/dev/null | head -20"
        exit_code, stdout, _ = await _run_command(cmd, workspace, timeout=5.0)

        if exit_code == 0 and stdout:
            for line in stdout.split("\n"):
                git_dir = Path(line.strip())
                if git_dir.exists() and git_dir.name == ".git":
                    repo_path = git_dir.parent
                    if repo_path not in discovered:
                        discovered.append(repo_path)

    return discovered


async def _filter_relevant_repos(
    all_repos: list[Path],
    workspace: Path,
    agent_name: str,
) -> list[Path]:
    """Use Haiku to filter discovered repos down to relevant ones.

    If no API keys are available, returns all repos in/under the workspace.
    """
    from chorus.sub_agents.runner import run_sub_agent

    if not all_repos:
        return []

    # Fallback: If no API keys, just return repos under workspace
    import os

    has_keys = bool(
        os.environ.get("ANTHROPIC_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )

    if not has_keys:
        logger.info("No API keys available, using all repos under workspace")
        return [r for r in all_repos if r == workspace or workspace in r.parents]

    # Prepare context
    repo_list = "\n".join(f"- {r}" for r in all_repos)
    user_content = (
        f"Agent name: {agent_name}\n"
        f"Workspace: {workspace}\n\n"
        f"Discovered repositories:\n{repo_list}"
    )

    messages = [{"role": "user", "content": user_content}]

    result = await run_sub_agent(
        system_prompt=_PATH_DISCOVERY_PROMPT,
        messages=messages,
        timeout=_DISCOVER_TIMEOUT,
    )

    if not result.success:
        logger.warning("Path discovery failed: %s — using workspace fallback", result.error)
        return [r for r in all_repos if r == workspace or workspace in r.parents]

    # Parse JSON response
    try:
        filtered_paths = json.loads(result.output)
        if not isinstance(filtered_paths, list):
            logger.warning("Path discovery returned non-list — using workspace fallback")
            return [r for r in all_repos if r == workspace or workspace in r.parents]

        resolved = []
        for p in filtered_paths:
            path = Path(p)
            if path.exists() and path.is_dir():
                resolved.append(path)

        if resolved:
            return resolved
        return [
            r for r in all_repos if r == workspace or workspace in r.parents
        ]

    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "Failed to parse path discovery JSON: %s — using workspace fallback",
            exc,
        )
        return [
            r for r in all_repos if r == workspace or workspace in r.parents
        ]


# ---------------------------------------------------------------------------
# Step 2: Static Script Execution
# ---------------------------------------------------------------------------


async def _collect_git_info(repo_path: Path) -> dict[str, Any]:
    """Collect git status info from a single repository.

    Returns a dict with keys:
    - success: bool
    - branch: str
    - commits: list[str] (last 3 commit messages)
    - changes: str (diffstat for working tree changes)
    - staged: str (diffstat for staged changes)
    - untracked_count: int
    - error: str | None
    """
    result: dict[str, Any] = {
        "success": False,
        "branch": "",
        "commits": [],
        "changes": "",
        "staged": "",
        "untracked_count": 0,
        "error": None,
    }

    # Check if it's a git repo
    if not (repo_path / ".git").exists():
        result["error"] = "Not a git repository"
        return result

    # Get current branch
    exit_code, stdout, stderr = await _run_command(
        "git branch --show-current",
        repo_path,
    )
    if exit_code != 0:
        result["error"] = f"git branch failed: {stderr}"
        return result
    result["branch"] = stdout or "(detached HEAD)"

    # Get last 3 commits
    exit_code, stdout, stderr = await _run_command(
        "git log --oneline -3",
        repo_path,
    )
    if exit_code == 0 and stdout:
        result["commits"] = stdout.split("\n")[:3]

    # Get working tree changes (diffstat)
    exit_code, stdout, stderr = await _run_command(
        "git diff --stat",
        repo_path,
    )
    if exit_code == 0:
        result["changes"] = stdout

    # Get staged changes (diffstat)
    exit_code, stdout, stderr = await _run_command(
        "git diff --cached --stat",
        repo_path,
    )
    if exit_code == 0:
        result["staged"] = stdout

    # Get untracked file count
    exit_code, stdout, stderr = await _run_command(
        "git ls-files --others --exclude-standard | wc -l",
        repo_path,
    )
    if exit_code == 0 and stdout:
        try:
            result["untracked_count"] = int(stdout)
        except ValueError:
            result["untracked_count"] = 0

    result["success"] = True
    return result


# ---------------------------------------------------------------------------
# Step 3: Error Recovery
# ---------------------------------------------------------------------------


async def _recover_failed_paths(
    failed: dict[Path, str],
) -> dict[Path, Path | None]:
    """Use Haiku to suggest corrected paths for failed repos.

    Returns a dict mapping old_path -> corrected_path (or None if no correction).
    """
    from chorus.sub_agents.runner import run_sub_agent

    if not failed:
        return {}

    # Build error context
    error_lines = [f"{path}: {error}" for path, error in failed.items()]
    user_content = "\n".join(error_lines)

    messages = [{"role": "user", "content": user_content}]

    result = await run_sub_agent(
        system_prompt=_ERROR_RECOVERY_PROMPT,
        messages=messages,
        timeout=_ERROR_RECOVERY_TIMEOUT,
    )

    if not result.success:
        logger.warning("Error recovery failed: %s", result.error)
        return {}

    # Parse JSON response
    try:
        corrections = json.loads(result.output)
        if not isinstance(corrections, dict):
            logger.warning("Error recovery returned non-dict")
            return {}

        resolved: dict[Path, Path | None] = {}
        for old_path_str, new_path_str in corrections.items():
            old_path = Path(old_path_str)
            if new_path_str is None:
                resolved[old_path] = None
            else:
                new_path = Path(new_path_str)
                if new_path.exists() and new_path.is_dir():
                    resolved[old_path] = new_path
                else:
                    resolved[old_path] = None

        return resolved

    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Failed to parse error recovery JSON: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Step 4: Format Output
# ---------------------------------------------------------------------------


def _format_git_report(results: dict[Path, dict[str, Any]]) -> str:
    """Format collected git info into a human-readable report."""
    if not results:
        return "No git repositories found or accessible."

    lines = []

    for repo_path, info in sorted(results.items()):
        lines.append(f"\n📁 {repo_path}")

        if not info["success"]:
            lines.append(f"   ❌ Error: {info['error']}")
            continue

        # Branch
        lines.append(f"   Branch: {info['branch']}")

        # Recent commits
        if info["commits"]:
            lines.append("   Recent commits:")
            for commit in info["commits"]:
                lines.append(f"     {commit}")
        else:
            lines.append("   Recent commits: (none)")

        # Working tree changes
        if info["changes"]:
            # Parse diffstat to show concise summary
            change_lines = info["changes"].split("\n")
            if len(change_lines) > 1:
                last = change_lines[-1]
                summary_line = (
                    last if last.strip() else change_lines[-2]
                )
                lines.append(f"   Changes: {summary_line}")
                # Show individual file stats (up to 5 files)
                for line in change_lines[:5]:
                    if line.strip() and "|" in line:
                        lines.append(f"     {line}")
            else:
                lines.append("   Changes: (clean)")
        else:
            lines.append("   Changes: (clean)")

        # Staged changes
        if info["staged"]:
            staged_lines = info["staged"].split("\n")
            if len(staged_lines) > 1:
                last = staged_lines[-1]
                summary_line = (
                    last if last.strip() else staged_lines[-2]
                )
                lines.append(f"   Staged: {summary_line}")
            else:
                lines.append("   Staged: (none)")
        else:
            lines.append("   Staged: (none)")

        # Untracked files
        if info["untracked_count"] > 0:
            lines.append(f"   Untracked: {info['untracked_count']} files")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main tool handler
# ---------------------------------------------------------------------------


async def git_status_execute(
    workspace: Path,
    agent_name: str,
    chorus_home: Path | None = None,
    host_execution: bool = False,
) -> str:
    """Get a comprehensive git status report for all relevant repositories.

    This tool discovers git repositories, filters them based on agent context,
    collects git status information, and formats it into a readable report.

    Parameters
    ----------
    workspace:
        Agent workspace directory (injected by tool loop).
    agent_name:
        Agent identifier (injected by tool loop).
    chorus_home:
        Chorus home directory (injected by tool loop).
    host_execution:
        Whether host execution is enabled (injected by tool loop).

    Returns
    -------
    str
        Formatted git status report for all relevant repositories.
    """
    try:
        # Overall timeout wrapper
        return await asyncio.wait_for(
            _git_status_execute_impl(workspace, agent_name, chorus_home, host_execution),
            timeout=_OVERALL_TIMEOUT,
        )
    except TimeoutError:
        return f"❌ Git status operation timed out after {_OVERALL_TIMEOUT}s"
    except Exception as exc:
        logger.exception("Git status execution failed")
        return f"❌ Git status failed: {type(exc).__name__}: {exc}"


async def _git_status_execute_impl(
    workspace: Path,
    agent_name: str,
    chorus_home: Path | None,
    host_execution: bool,
) -> str:
    """Implementation of git_status_execute (inner function for timeout wrapper)."""
    # Step 1: Discover all git repos
    logger.info("Discovering git repositories...")
    all_repos = await _discover_git_repos(workspace)

    if not all_repos:
        return "No git repositories found."

    logger.info("Discovered %d git repositories", len(all_repos))

    # Step 2: Filter to relevant repos using Haiku
    logger.info("Filtering relevant repositories...")
    relevant_repos = await _filter_relevant_repos(all_repos, workspace, agent_name)

    if not relevant_repos:
        return "No relevant git repositories found for this agent."

    logger.info("Filtered to %d relevant repositories", len(relevant_repos))

    # Step 3: Collect git info from each repo
    results: dict[Path, dict[str, Any]] = {}
    failed: dict[Path, str] = {}

    tasks = [_collect_git_info(repo) for repo in relevant_repos]
    collected = await asyncio.gather(*tasks, return_exceptions=True)

    for repo, info in zip(relevant_repos, collected, strict=False):
        if isinstance(info, BaseException):
            failed[repo] = str(info)
            logger.warning("Failed to collect git info from %s: %s", repo, info)
        elif not info["success"]:
            failed[repo] = info["error"] or "Unknown error"
        else:
            results[repo] = info

    # Step 4: Error recovery (if any failed)
    if failed:
        logger.info("Attempting error recovery for %d failed repos", len(failed))
        corrections = await _recover_failed_paths(failed)

        # Retry with corrected paths
        for old_path, new_path in corrections.items():
            if new_path is not None and new_path not in results:
                info = await _collect_git_info(new_path)
                if info["success"]:
                    results[new_path] = info
                    failed.pop(old_path, None)
                    logger.info("Recovered %s -> %s", old_path, new_path)

    # Add failed repos to results for reporting
    for repo, error in failed.items():
        results[repo] = {
            "success": False,
            "error": error,
            "branch": "",
            "commits": [],
            "changes": "",
            "staged": "",
            "untracked_count": 0,
        }

    # Step 5: Format output
    return _format_git_report(results)
