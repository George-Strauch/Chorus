"""Git operations for agent workspaces.

All commands delegate to :func:`chorus.tools.bash.bash_execute` — no Python
git library is used.  Permission checks use action strings of the form
``tool:git:<operation> <args>``.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from chorus.permissions.engine import (
    PermissionProfile,
    PermissionResult,
    check,
    format_action,
)
from chorus.tools.bash import (
    CommandDeniedError,
    CommandNeedsApprovalError,
    bash_execute,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GitError(Exception):
    """Base error for git operations."""


class UnsupportedForgeError(GitError):
    """Raised when the remote URL does not match a known forge."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GitResult:
    """Structured result from a git operation."""

    operation: str
    success: bool
    stdout: str
    stderr: str
    commit_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "commit_hash": self.commit_hash,
        }


# ---------------------------------------------------------------------------
# Internal open profile for inner bash calls (avoids double permission check)
# ---------------------------------------------------------------------------

_OPEN_PROFILE = PermissionProfile(allow=[".*"], ask=[])

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

_COMMIT_HASH_RE = re.compile(r"\[[\w/.-]+\s+([0-9a-f]{7,40})\]")


async def _git(
    workspace: Path,
    args: str,
    profile: PermissionProfile,
    operation: str,
    host_execution: bool = False,
    scope_path: Path | None = None,
) -> GitResult:
    """Run ``git <args>`` with permission checking.

    Parameters
    ----------
    workspace:
        Agent workspace directory.
    args:
        Arguments to pass to git (e.g. ``"commit -m 'msg'"``).
    profile:
        Permission profile to check against.
    operation:
        Logical operation name for the action string (e.g. ``"push"``).
    host_execution:
        Whether to use the full host environment.
    scope_path:
        Host filesystem path for credential resolution auto-detection.

    Raises
    ------
    CommandDeniedError
        If the permission engine denies the action.
    CommandNeedsApprovalError
        If the permission engine returns ASK.
    """
    action = format_action("git", f"{operation} {args}".strip())
    perm_result = check(action, profile)
    if perm_result is PermissionResult.DENY:
        raise CommandDeniedError(f"Permission denied: {action}")
    if perm_result is PermissionResult.ASK:
        raise CommandNeedsApprovalError(f"git {args}", f"Needs approval: {action}")

    result = await bash_execute(
        f"git {args}",
        workspace,
        profile=_OPEN_PROFILE,
        timeout=60.0,
        host_execution=host_execution,
        scope_path=scope_path,
    )

    return GitResult(
        operation=operation,
        success=result.exit_code == 0,
        stdout=result.stdout,
        stderr=result.stderr,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def git_init(
    workspace: Path,
    agent_name: str,
    profile: PermissionProfile,
    host_execution: bool = False,
    scope_path: Path | None = None,
) -> GitResult:
    """Initialize a git repo in *workspace* and set user config."""
    result = await _git(workspace, "init", profile, "init",
                        host_execution=host_execution, scope_path=scope_path)
    if not result.success:
        return result
    await _git(
        workspace,
        f"config user.name {shlex.quote(agent_name)}",
        profile,
        "config",
        host_execution=host_execution,
        scope_path=scope_path,
    )
    await _git(
        workspace,
        f"config user.email {shlex.quote(agent_name + '@chorus.local')}",
        profile,
        "config",
        host_execution=host_execution,
        scope_path=scope_path,
    )
    return result


async def git_commit(
    workspace: Path,
    message: str,
    profile: PermissionProfile,
    files: list[str] | None = None,
    host_execution: bool = False,
    scope_path: Path | None = None,
) -> GitResult:
    """Stage files and commit.

    If *files* is given, only those files are staged; otherwise ``git add -A``.
    Returns a :class:`GitResult` with ``commit_hash`` set on success.
    """
    if files:
        for f in files:
            await _git(workspace, f"add {shlex.quote(f)}", profile, "add",
                        host_execution=host_execution, scope_path=scope_path)
    else:
        await _git(workspace, "add -A", profile, "add",
                    host_execution=host_execution, scope_path=scope_path)

    result = await _git(
        workspace,
        f"commit -m {shlex.quote(message)}",
        profile,
        "commit",
        host_execution=host_execution,
        scope_path=scope_path,
    )

    if result.success:
        m = _COMMIT_HASH_RE.search(result.stdout)
        if m:
            result.commit_hash = m.group(1)

    return result


async def git_push(
    workspace: Path,
    remote: str,
    branch: str,
    profile: PermissionProfile,
    host_execution: bool = False,
    scope_path: Path | None = None,
) -> GitResult:
    """Push to a remote."""
    return await _git(
        workspace,
        f"push {shlex.quote(remote)} {shlex.quote(branch)}",
        profile,
        "push",
        host_execution=host_execution,
        scope_path=scope_path,
    )


async def git_branch(
    workspace: Path,
    profile: PermissionProfile,
    branch_name: str | None = None,
    delete: bool = False,
    host_execution: bool = False,
    scope_path: Path | None = None,
) -> GitResult:
    """Create, list, or delete branches."""
    if branch_name is None:
        return await _git(workspace, "branch", profile, "branch",
                          host_execution=host_execution, scope_path=scope_path)
    if delete:
        return await _git(
            workspace,
            f"branch -d {shlex.quote(branch_name)}",
            profile,
            "branch",
            host_execution=host_execution,
            scope_path=scope_path,
        )
    return await _git(
        workspace,
        f"branch {shlex.quote(branch_name)}",
        profile,
        "branch",
        host_execution=host_execution,
        scope_path=scope_path,
    )


async def git_checkout(
    workspace: Path,
    ref: str,
    profile: PermissionProfile,
    create: bool = False,
    host_execution: bool = False,
    scope_path: Path | None = None,
) -> GitResult:
    """Checkout a branch, tag, or commit."""
    flag = "-b " if create else ""
    return await _git(
        workspace,
        f"checkout {flag}{shlex.quote(ref)}",
        profile,
        "checkout",
        host_execution=host_execution,
        scope_path=scope_path,
    )


async def git_diff(
    workspace: Path,
    profile: PermissionProfile,
    ref1: str | None = None,
    ref2: str | None = None,
    host_execution: bool = False,
    scope_path: Path | None = None,
) -> GitResult:
    """Show diff — working tree vs HEAD, or between two refs."""
    if ref1 and ref2:
        args = f"diff {shlex.quote(ref1)} {shlex.quote(ref2)}"
    elif ref1:
        args = f"diff {shlex.quote(ref1)}"
    else:
        args = "diff"
    return await _git(workspace, args, profile, "diff",
                      host_execution=host_execution, scope_path=scope_path)


async def git_log(
    workspace: Path,
    profile: PermissionProfile,
    count: int = 20,
    oneline: bool = False,
    host_execution: bool = False,
    scope_path: Path | None = None,
) -> GitResult:
    """Show commit log."""
    fmt = " --oneline" if oneline else ""
    return await _git(workspace, f"log -n {count}{fmt}", profile, "log",
                      host_execution=host_execution, scope_path=scope_path)


# ---------------------------------------------------------------------------
# Merge request / pull request
# ---------------------------------------------------------------------------


async def _detect_forge(
    workspace: Path,
    host_execution: bool = False,
    scope_path: Path | None = None,
) -> str:
    """Detect the forge type from the origin remote URL.

    Returns ``"github"`` or ``"gitlab"``.

    Raises
    ------
    GitError
        If there is no origin remote.
    UnsupportedForgeError
        If the remote URL doesn't match a known forge.
    """
    result = await bash_execute(
        "git remote get-url origin",
        workspace,
        profile=_OPEN_PROFILE,
        timeout=10.0,
        host_execution=host_execution,
        scope_path=scope_path,
    )
    if result.exit_code != 0:
        raise GitError(f"No origin remote configured: {result.stderr.strip()}")

    url = result.stdout.strip()
    if "github.com" in url:
        return "github"
    if "gitlab" in url:
        return "gitlab"
    raise UnsupportedForgeError(f"Unsupported forge for remote URL: {url}")


async def git_merge_request(
    workspace: Path,
    title: str,
    description: str,
    source_branch: str,
    target_branch: str,
    profile: PermissionProfile,
    host_execution: bool = False,
    scope_path: Path | None = None,
) -> GitResult:
    """Create a merge/pull request on the detected forge.

    Uses ``gh pr create`` for GitHub, ``glab mr create`` for GitLab.
    """
    # Permission check first
    action = format_action("git", f"merge_request {source_branch} -> {target_branch}")
    perm_result = check(action, profile)
    if perm_result is PermissionResult.DENY:
        raise CommandDeniedError(f"Permission denied: {action}")
    if perm_result is PermissionResult.ASK:
        raise CommandNeedsApprovalError(
            f"merge_request {source_branch} -> {target_branch}",
            f"Needs approval: {action}",
        )

    forge = await _detect_forge(workspace, host_execution=host_execution, scope_path=scope_path)

    q_title = shlex.quote(title)
    q_desc = shlex.quote(description)
    q_src = shlex.quote(source_branch)
    q_tgt = shlex.quote(target_branch)

    if forge == "github":
        cmd = f"gh pr create --title {q_title} --body {q_desc} --head {q_src} --base {q_tgt}"
    else:
        cmd = (
            f"glab mr create --title {q_title} --description {q_desc}"
            f" --source-branch {q_src} --target-branch {q_tgt}"
        )

    result = await bash_execute(
        cmd, workspace, profile=_OPEN_PROFILE, timeout=30.0,
        host_execution=host_execution, scope_path=scope_path,
    )

    return GitResult(
        operation="merge_request",
        success=result.exit_code == 0,
        stdout=result.stdout,
        stderr=result.stderr,
    )
