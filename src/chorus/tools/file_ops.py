"""File operation tools for agent workspaces — create, str_replace, view.

All paths are resolved relative to the agent's workspace root.
Path traversal is prevented by resolving symlinks then checking
the result is within the workspace jail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PathTraversalError(Exception):
    """Raised when a resolved path escapes the workspace root."""


class StringNotFoundError(Exception):
    """Raised when str_replace cannot find the target string."""


class AmbiguousMatchError(Exception):
    """Raised when str_replace finds the target string more than once."""


class BinaryFileError(Exception):
    """Raised when attempting to view a binary file."""


class FileNotFoundInWorkspaceError(FileNotFoundError):
    """Raised when the target file does not exist in the workspace."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class FileResult:
    """Structured result returned by file tools."""

    path: str
    action: str
    success: bool
    content_snippet: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "path": self.path,
            "action": self.action,
            "success": self.success,
        }
        if self.content_snippet is not None:
            d["content_snippet"] = self.content_snippet
        if self.error is not None:
            d["error"] = self.error
        return d


# ---------------------------------------------------------------------------
# Path jail
# ---------------------------------------------------------------------------


def resolve_in_workspace(workspace: Path, relative_path: str) -> Path:
    """Resolve *relative_path* inside *workspace*, raising on escape.

    Symlinks are resolved **before** the containment check so they
    cannot be used to break out.
    """
    resolved = (workspace / relative_path).resolve()
    workspace_resolved = workspace.resolve()
    if resolved == workspace_resolved:
        return resolved
    if not str(resolved).startswith(str(workspace_resolved) + "/"):
        raise PathTraversalError(
            f"Path {relative_path!r} resolves outside workspace"
        )
    return resolved


# ---------------------------------------------------------------------------
# create_file
# ---------------------------------------------------------------------------


async def create_file(
    workspace: Path,
    path: str,
    content: str,
) -> FileResult:
    """Create (or overwrite) a file inside the workspace."""
    resolved = resolve_in_workspace(workspace, path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return FileResult(path=path, action="created", success=True)


# ---------------------------------------------------------------------------
# str_replace
# ---------------------------------------------------------------------------

_CONTEXT_LINES = 3


async def str_replace(
    workspace: Path,
    path: str,
    old_str: str,
    new_str: str,
) -> FileResult:
    """Replace exactly one occurrence of *old_str* with *new_str* in a file."""
    resolved = resolve_in_workspace(workspace, path)

    if not resolved.exists():
        raise FileNotFoundInWorkspaceError(f"File not found: {path}")

    content = resolved.read_text(encoding="utf-8")
    count = content.count(old_str)

    if count == 0:
        raise StringNotFoundError(
            f"String not found in {path}: {old_str!r}"
        )
    if count > 1:
        raise AmbiguousMatchError(
            f"String appears {count} times in {path} — must be unique"
        )

    original_mode = resolved.stat().st_mode
    new_content = content.replace(old_str, new_str, 1)
    resolved.write_text(new_content, encoding="utf-8")
    resolved.chmod(original_mode)

    # Build context snippet around the replacement
    snippet = _context_around(new_content, new_str)

    return FileResult(
        path=path,
        action="str_replace",
        success=True,
        content_snippet=snippet,
    )


def _context_around(content: str, target: str) -> str:
    """Return lines around the first occurrence of *target* with line numbers."""
    lines = content.splitlines()
    target_line = None
    for i, line in enumerate(lines):
        if target in line:
            target_line = i
            break

    if target_line is None:
        # target might span lines or be empty — return first few lines
        start = 0
        end = min(len(lines), _CONTEXT_LINES * 2 + 1)
    else:
        start = max(0, target_line - _CONTEXT_LINES)
        end = min(len(lines), target_line + _CONTEXT_LINES + 1)

    numbered = [f"{i + 1}\t{lines[i]}" for i in range(start, end)]
    return "\n".join(numbered)


# ---------------------------------------------------------------------------
# view
# ---------------------------------------------------------------------------

_BINARY_CHECK_SIZE = 8192


async def view(
    workspace: Path,
    path: str,
    offset: int | None = None,
    limit: int | None = None,
) -> FileResult:
    """View a file's contents with line numbers.

    Parameters
    ----------
    offset:
        1-based line number to start from (default: 1).
    limit:
        Number of lines to return (default: all).
    """
    resolved = resolve_in_workspace(workspace, path)

    if not resolved.exists():
        raise FileNotFoundInWorkspaceError(f"File not found: {path}")

    if resolved.is_dir():
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in resolved.iterdir())
        listing = "\n".join(entries) if entries else "(empty directory)"
        return FileResult(
            path=path,
            action="view",
            success=True,
            content_snippet=f"Directory listing of {path}/:\n{listing}",
        )

    # Binary check
    raw = resolved.read_bytes()
    check_chunk = raw[:_BINARY_CHECK_SIZE]
    if b"\x00" in check_chunk:
        raise BinaryFileError(f"File appears to be binary: {path}")

    content = raw.decode("utf-8")
    lines = content.splitlines()

    # Apply offset (1-based) and limit
    start = (offset - 1) if offset is not None else 0
    start = max(0, start)
    end = start + limit if limit is not None else len(lines)

    selected = lines[start:end]
    numbered = [f"{start + i + 1}\t{line}" for i, line in enumerate(selected)]
    snippet = "\n".join(numbered)

    return FileResult(
        path=path,
        action="view",
        success=True,
        content_snippet=snippet,
    )
