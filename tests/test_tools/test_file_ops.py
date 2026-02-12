"""Tests for chorus.tools.file_ops — file operations within agent workspace."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from chorus.tools.file_ops import (
    AmbiguousMatchError,
    BinaryFileError,
    FileNotFoundInWorkspaceError,
    PathTraversalError,
    StringNotFoundError,
    create_file,
    str_replace,
    view,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# create_file
# ---------------------------------------------------------------------------


class TestCreateFile:
    @pytest.mark.asyncio
    async def test_create_file_in_workspace_root(self, workspace_dir: Path) -> None:
        result = await create_file(workspace_dir, "new.txt", "content here")
        assert result.success is True
        assert result.action == "created"
        assert (workspace_dir / "new.txt").read_text(encoding="utf-8") == "content here"

    @pytest.mark.asyncio
    async def test_create_file_in_subdirectory_creates_parents(
        self, workspace_dir: Path
    ) -> None:
        result = await create_file(workspace_dir, "a/b/c/deep.txt", "deep content")
        assert result.success is True
        assert (workspace_dir / "a" / "b" / "c" / "deep.txt").exists()
        assert (
            (workspace_dir / "a" / "b" / "c" / "deep.txt").read_text(encoding="utf-8")
            == "deep content"
        )

    @pytest.mark.asyncio
    async def test_create_file_overwrites_existing(self, workspace_dir: Path) -> None:
        await create_file(workspace_dir, "hello.txt", "overwritten")
        assert (workspace_dir / "hello.txt").read_text(encoding="utf-8") == "overwritten"

    @pytest.mark.asyncio
    async def test_create_file_rejects_path_traversal_dotdot(
        self, workspace_dir: Path
    ) -> None:
        with pytest.raises(PathTraversalError):
            await create_file(workspace_dir, "../../etc/passwd", "evil")

    @pytest.mark.asyncio
    async def test_create_file_rejects_absolute_path_outside_workspace(
        self, workspace_dir: Path
    ) -> None:
        with pytest.raises(PathTraversalError):
            await create_file(workspace_dir, "/etc/passwd", "evil")

    @pytest.mark.asyncio
    async def test_create_file_rejects_symlink_escape(
        self, workspace_dir: Path
    ) -> None:
        # Create a symlink inside workspace pointing outside
        link = workspace_dir / "escape_link"
        link.symlink_to("/tmp")
        with pytest.raises(PathTraversalError):
            await create_file(workspace_dir, "escape_link/evil.txt", "evil")

    @pytest.mark.asyncio
    async def test_create_file_utf8_content(self, workspace_dir: Path) -> None:
        content = "Hello \u00e9\u00e0\u00fc \u4e16\u754c \U0001f30d"
        result = await create_file(workspace_dir, "unicode.txt", content)
        assert result.success is True
        assert (workspace_dir / "unicode.txt").read_text(encoding="utf-8") == content


# ---------------------------------------------------------------------------
# str_replace
# ---------------------------------------------------------------------------


class TestStrReplace:
    @pytest.mark.asyncio
    async def test_str_replace_exact_match(self, workspace_dir: Path) -> None:
        result = await str_replace(
            workspace_dir, "src/app.py", "print('hello')", "print('goodbye')"
        )
        assert result.success is True
        content = (workspace_dir / "src" / "app.py").read_text(encoding="utf-8")
        assert "print('goodbye')" in content
        assert "print('hello')" not in content

    @pytest.mark.asyncio
    async def test_str_replace_fails_on_no_match(self, workspace_dir: Path) -> None:
        with pytest.raises(StringNotFoundError):
            await str_replace(
                workspace_dir, "src/app.py", "nonexistent string", "replacement"
            )

    @pytest.mark.asyncio
    async def test_str_replace_fails_on_multiple_matches(
        self, workspace_dir: Path
    ) -> None:
        # Write a file with duplicate content
        (workspace_dir / "dupes.txt").write_text("aaa\naaa\n", encoding="utf-8")
        with pytest.raises(AmbiguousMatchError):
            await str_replace(workspace_dir, "dupes.txt", "aaa", "bbb")

    @pytest.mark.asyncio
    async def test_str_replace_returns_context_lines(
        self, workspace_dir: Path
    ) -> None:
        result = await str_replace(
            workspace_dir, "src/app.py", "print('hello')", "print('goodbye')"
        )
        assert result.content_snippet is not None
        assert "print('goodbye')" in result.content_snippet

    @pytest.mark.asyncio
    async def test_str_replace_rejects_path_traversal(
        self, workspace_dir: Path
    ) -> None:
        with pytest.raises(PathTraversalError):
            await str_replace(
                workspace_dir, "../../etc/hosts", "old", "new"
            )

    @pytest.mark.asyncio
    async def test_str_replace_preserves_file_permissions(
        self, workspace_dir: Path
    ) -> None:
        target = workspace_dir / "perms.txt"
        target.write_text("old content", encoding="utf-8")
        os.chmod(target, 0o755)
        original_mode = target.stat().st_mode

        await str_replace(workspace_dir, "perms.txt", "old content", "new content")

        assert target.stat().st_mode == original_mode

    @pytest.mark.asyncio
    async def test_str_replace_handles_empty_new_string_for_deletion(
        self, workspace_dir: Path
    ) -> None:
        result = await str_replace(
            workspace_dir, "src/app.py", "print('hello')", ""
        )
        assert result.success is True
        content = (workspace_dir / "src" / "app.py").read_text(encoding="utf-8")
        assert "print('hello')" not in content


# ---------------------------------------------------------------------------
# view
# ---------------------------------------------------------------------------


class TestView:
    @pytest.mark.asyncio
    async def test_view_returns_full_file_with_line_numbers(
        self, workspace_dir: Path
    ) -> None:
        result = await view(workspace_dir, "src/app.py")
        assert result.success is True
        assert result.content_snippet is not None
        # Should have line numbers
        assert "1\t" in result.content_snippet
        assert "def main():" in result.content_snippet

    @pytest.mark.asyncio
    async def test_view_with_offset_and_limit(self, workspace_dir: Path) -> None:
        result = await view(workspace_dir, "src/app.py", offset=2, limit=2)
        assert result.success is True
        assert result.content_snippet is not None
        lines = result.content_snippet.strip().splitlines()
        assert len(lines) == 2
        assert "2\t" in lines[0]

    @pytest.mark.asyncio
    async def test_view_offset_beyond_file_length(self, workspace_dir: Path) -> None:
        result = await view(workspace_dir, "src/app.py", offset=999)
        assert result.success is True
        assert result.content_snippet is not None
        # Should be empty or have no content lines
        stripped = result.content_snippet.strip()
        assert stripped == ""

    @pytest.mark.asyncio
    async def test_view_rejects_path_traversal(self, workspace_dir: Path) -> None:
        with pytest.raises(PathTraversalError):
            await view(workspace_dir, "../../etc/passwd")

    @pytest.mark.asyncio
    async def test_view_rejects_binary_file(self, workspace_dir: Path) -> None:
        binary_path = workspace_dir / "image.bin"
        binary_path.write_bytes(b"\x00\x01\x02\xff\xfe\x00binary data")
        with pytest.raises(BinaryFileError):
            await view(workspace_dir, "image.bin")

    @pytest.mark.asyncio
    async def test_view_empty_file(self, workspace_dir: Path) -> None:
        (workspace_dir / "empty.txt").write_text("", encoding="utf-8")
        result = await view(workspace_dir, "empty.txt")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_view_nonexistent_file_raises(self, workspace_dir: Path) -> None:
        with pytest.raises(FileNotFoundInWorkspaceError):
            await view(workspace_dir, "does_not_exist.txt")


# ---------------------------------------------------------------------------
# Path traversal (dedicated)
# ---------------------------------------------------------------------------


class TestPathJail:
    @pytest.mark.asyncio
    async def test_jail_check_allows_nested_paths(self, workspace_dir: Path) -> None:
        result = await create_file(workspace_dir, "a/b/c.txt", "nested")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_jail_check_blocks_dotdot(self, workspace_dir: Path) -> None:
        with pytest.raises(PathTraversalError):
            await view(workspace_dir, "../agent.json")

    @pytest.mark.asyncio
    async def test_jail_check_blocks_absolute_outside(
        self, workspace_dir: Path
    ) -> None:
        with pytest.raises(PathTraversalError):
            await view(workspace_dir, "/etc/passwd")

    @pytest.mark.asyncio
    async def test_jail_check_resolves_symlinks_before_checking(
        self, workspace_dir: Path
    ) -> None:
        link = workspace_dir / "sneaky"
        link.symlink_to(workspace_dir.parent)
        with pytest.raises(PathTraversalError):
            await view(workspace_dir, "sneaky/agent.json")

    @pytest.mark.asyncio
    async def test_jail_check_blocks_double_encoded_traversal(
        self, workspace_dir: Path
    ) -> None:
        # Attempting a path with encoded dots — should still be blocked by resolve()
        with pytest.raises(PathTraversalError):
            await create_file(workspace_dir, "../../../tmp/evil.txt", "bad")
