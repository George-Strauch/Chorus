"""Tests for chorus.tools.file_ops — file operations within agent workspace."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from chorus.tools.file_ops import (
    AmbiguousMatchError,
    BinaryFileError,
    FileNotFoundInWorkspaceError,
    StringNotFoundError,
    append_file,
    create_file,
    resolve_path,
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
    async def test_create_file_with_absolute_path(self, tmp_path: Path) -> None:
        target = tmp_path / "outside" / "abs.txt"
        result = await create_file(tmp_path / "workspace", str(target), "absolute")
        assert result.success is True
        assert target.read_text(encoding="utf-8") == "absolute"

    @pytest.mark.asyncio
    async def test_create_file_dotdot_resolves_outside_workspace(
        self, workspace_dir: Path
    ) -> None:
        result = await create_file(workspace_dir, "../sibling.txt", "outside")
        assert result.success is True
        assert (workspace_dir.parent / "sibling.txt").read_text(encoding="utf-8") == "outside"

    @pytest.mark.asyncio
    async def test_create_file_utf8_content(self, workspace_dir: Path) -> None:
        content = "Hello \u00e9\u00e0\u00fc \u4e16\u754c \U0001f30d"
        result = await create_file(workspace_dir, "unicode.txt", content)
        assert result.success is True
        assert (workspace_dir / "unicode.txt").read_text(encoding="utf-8") == content


# ---------------------------------------------------------------------------
# append_file
# ---------------------------------------------------------------------------


class TestAppendFile:
    @pytest.mark.asyncio
    async def test_append_to_existing_file(self, workspace_dir: Path) -> None:
        f = workspace_dir / "out.txt"
        f.write_text("line 1\n", encoding="utf-8")

        result = await append_file(workspace_dir, "out.txt", "line 2\n")
        assert result.success
        assert result.action == "appended"
        assert f.read_text(encoding="utf-8") == "line 1\nline 2\n"

    @pytest.mark.asyncio
    async def test_append_creates_file_if_missing(self, workspace_dir: Path) -> None:
        result = await append_file(workspace_dir, "new.txt", "hello")
        assert result.success
        assert (workspace_dir / "new.txt").read_text(encoding="utf-8") == "hello"

    @pytest.mark.asyncio
    async def test_append_creates_directories(self, workspace_dir: Path) -> None:
        result = await append_file(workspace_dir, "sub/dir/file.txt", "content")
        assert result.success
        assert (workspace_dir / "sub/dir/file.txt").read_text(encoding="utf-8") == "content"

    @pytest.mark.asyncio
    async def test_multiple_appends(self, workspace_dir: Path) -> None:
        await append_file(workspace_dir, "build.txt", "part 1\n")
        await append_file(workspace_dir, "build.txt", "part 2\n")
        await append_file(workspace_dir, "build.txt", "part 3\n")

        content = (workspace_dir / "build.txt").read_text(encoding="utf-8")
        assert content == "part 1\npart 2\npart 3\n"


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
    async def test_str_replace_with_absolute_path(self, tmp_path: Path) -> None:
        target = tmp_path / "outside.txt"
        target.write_text("old content", encoding="utf-8")
        result = await str_replace(tmp_path / "workspace", str(target), "old", "new")
        assert result.success is True
        assert "new content" in target.read_text(encoding="utf-8")

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
    async def test_view_with_absolute_path(self, tmp_path: Path) -> None:
        target = tmp_path / "outside.txt"
        target.write_text("viewable content", encoding="utf-8")
        result = await view(tmp_path / "workspace", str(target))
        assert result.success is True
        assert "viewable content" in result.content_snippet

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

    @pytest.mark.asyncio
    async def test_view_directory_returns_listing(self, workspace_dir: Path) -> None:
        result = await view(workspace_dir, "src")
        assert result.success is True
        assert result.content_snippet is not None
        assert "app.py" in result.content_snippet
        assert "Directory listing" in result.content_snippet

    @pytest.mark.asyncio
    async def test_view_empty_directory(self, workspace_dir: Path) -> None:
        (workspace_dir / "emptydir").mkdir()
        result = await view(workspace_dir, "emptydir")
        assert result.success is True
        assert "(empty directory)" in result.content_snippet


# ---------------------------------------------------------------------------
# Path traversal (dedicated)
# ---------------------------------------------------------------------------


class TestResolvePath:
    @pytest.mark.asyncio
    async def test_nested_relative_paths_resolve_in_workspace(
        self, workspace_dir: Path
    ) -> None:
        result = await create_file(workspace_dir, "a/b/c.txt", "nested")
        assert result.success is True

    def test_resolve_path_relative(self, workspace_dir: Path) -> None:
        resolved = resolve_path(workspace_dir, "foo/bar.txt")
        assert resolved == (workspace_dir / "foo" / "bar.txt").resolve()

    def test_resolve_path_absolute(self, workspace_dir: Path) -> None:
        resolved = resolve_path(workspace_dir, "/tmp/some/file.txt")
        assert resolved == Path("/tmp/some/file.txt").resolve()

    def test_resolve_path_dotdot_relative(self, workspace_dir: Path) -> None:
        resolved = resolve_path(workspace_dir, "../sibling.txt")
        assert resolved == (workspace_dir.parent / "sibling.txt").resolve()
