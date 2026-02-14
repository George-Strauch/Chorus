"""Tests for chorus.agent.self_edit — agent self-modification."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from chorus.agent.self_edit import (
    SelfEditResult,
    _resolve_short_model_name,
    edit_docs,
    edit_model,
    edit_permissions,
    edit_system_prompt,
)
from chorus.permissions.engine import check, format_action, get_preset
from chorus.storage.db import Database

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(tmp_path: Path) -> AsyncGenerator[Database, None]:
    """Temporary database for audit log tests."""
    d = Database(tmp_path / "db" / "chorus.db")
    await d.init()
    yield d
    await d.close()


@pytest.fixture
def agent_dir(tmp_path: Path) -> Path:
    """Create a minimal agent directory with agent.json, docs/, workspace/."""
    adir = tmp_path / "agents" / "test-agent"
    adir.mkdir(parents=True)
    (adir / "workspace").mkdir()
    (adir / "docs").mkdir()
    agent_json = {
        "name": "test-agent",
        "channel_id": 123,
        "model": None,
        "system_prompt": "Original prompt.",
        "permissions": "standard",
    }
    (adir / "agent.json").write_text(json.dumps(agent_json, indent=4))
    return adir


@pytest.fixture
def workspace(agent_dir: Path) -> Path:
    return agent_dir / "workspace"


# ---------------------------------------------------------------------------
# SelfEditResult
# ---------------------------------------------------------------------------


class TestSelfEditResult:
    def test_to_dict(self) -> None:
        r = SelfEditResult(
            success=True,
            edit_type="system_prompt",
            old_value="old",
            new_value="new",
            message="Done",
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["edit_type"] == "system_prompt"
        assert "error" not in d

    def test_to_dict_with_error(self) -> None:
        r = SelfEditResult(
            success=False,
            edit_type="model",
            old_value="",
            new_value="bad",
            message="Failed",
            error="unavailable",
        )
        d = r.to_dict()
        assert d["error"] == "unavailable"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


class TestSelfEditSystemPrompt:
    @pytest.mark.asyncio
    async def test_updates_agent_json(self, workspace: Path) -> None:
        result = await edit_system_prompt(
            "New prompt.",
            workspace=workspace,
            agent_name="test-agent",
        )
        assert result.success is True
        data = json.loads((workspace.parent / "agent.json").read_text())
        assert data["system_prompt"] == "New prompt."

    @pytest.mark.asyncio
    async def test_old_value_returned(self, workspace: Path) -> None:
        result = await edit_system_prompt(
            "New prompt.",
            workspace=workspace,
            agent_name="test-agent",
        )
        assert result.old_value == "Original prompt."
        assert result.new_value == "New prompt."

    @pytest.mark.asyncio
    async def test_empty_string_rejected(self, workspace: Path) -> None:
        result = await edit_system_prompt(
            "   ",
            workspace=workspace,
            agent_name="test-agent",
        )
        assert result.success is False
        assert result.error == "empty_prompt"

    @pytest.mark.asyncio
    async def test_logs_to_audit(self, workspace: Path, db: Database) -> None:
        await edit_system_prompt(
            "New prompt.",
            workspace=workspace,
            agent_name="test-agent",
            db=db,
        )
        async with db.connection.execute(
            "SELECT action_string, detail FROM audit_log WHERE agent_name = ?",
            ("test-agent",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "tool:self_edit:system_prompt"
        detail = json.loads(row[1])
        assert detail["old_value"] == "Original prompt."
        assert detail["new_value"] == "New prompt."

    def test_permission_action_string(self) -> None:
        action = format_action("self_edit", "system_prompt")
        assert action == "tool:self_edit:system_prompt"
        profile = get_preset("standard")
        result = check(action, profile)
        from chorus.permissions.engine import PermissionResult

        assert result is PermissionResult.ASK


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------


class TestSelfEditDocs:
    @pytest.mark.asyncio
    async def test_creates_new_file(self, workspace: Path) -> None:
        result = await edit_docs(
            "guide.md",
            "# Guide\nSome content.",
            workspace=workspace,
            agent_name="test-agent",
        )
        assert result.success is True
        assert (workspace.parent / "docs" / "guide.md").read_text() == "# Guide\nSome content."

    @pytest.mark.asyncio
    async def test_updates_existing_file(self, workspace: Path) -> None:
        # Create initial
        (workspace.parent / "docs" / "notes.md").write_text("old content")
        result = await edit_docs(
            "notes.md",
            "new content",
            workspace=workspace,
            agent_name="test-agent",
        )
        assert result.success is True
        assert result.old_value == "old content"
        assert (workspace.parent / "docs" / "notes.md").read_text() == "new content"

    @pytest.mark.asyncio
    async def test_path_traversal_prevented(self, workspace: Path) -> None:
        result = await edit_docs(
            "../../etc/passwd",
            "hacked",
            workspace=workspace,
            agent_name="test-agent",
        )
        assert result.success is False
        assert result.error == "path_traversal"

    @pytest.mark.asyncio
    async def test_subdirectory_creation(self, workspace: Path) -> None:
        result = await edit_docs(
            "guides/setup.md",
            "# Setup",
            workspace=workspace,
            agent_name="test-agent",
        )
        assert result.success is True
        assert (workspace.parent / "docs" / "guides" / "setup.md").exists()

    def test_permission_action_string(self) -> None:
        action = format_action("self_edit", "docs README.md")
        assert action == "tool:self_edit:docs README.md"
        profile = get_preset("standard")
        result = check(action, profile)
        from chorus.permissions.engine import PermissionResult

        assert result is PermissionResult.ALLOW

    @pytest.mark.asyncio
    async def test_logs_to_audit(self, workspace: Path, db: Database) -> None:
        await edit_docs(
            "file.md",
            "content",
            workspace=workspace,
            agent_name="test-agent",
            db=db,
        )
        async with db.connection.execute(
            "SELECT action_string FROM audit_log WHERE agent_name = ?",
            ("test-agent",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "tool:self_edit:docs"


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------


class TestSelfEditPermissions:
    @pytest.mark.asyncio
    async def test_to_standard(self, workspace: Path) -> None:
        result = await edit_permissions(
            "standard",
            workspace=workspace,
            agent_name="test-agent",
        )
        assert result.success is True
        data = json.loads((workspace.parent / "agent.json").read_text())
        assert data["permissions"] == "standard"

    @pytest.mark.asyncio
    async def test_to_locked(self, workspace: Path) -> None:
        result = await edit_permissions(
            "locked",
            workspace=workspace,
            agent_name="test-agent",
        )
        assert result.success is True
        data = json.loads((workspace.parent / "agent.json").read_text())
        assert data["permissions"] == "locked"

    @pytest.mark.asyncio
    async def test_to_open_requires_admin(self, workspace: Path) -> None:
        result = await edit_permissions(
            "open",
            workspace=workspace,
            agent_name="test-agent",
            is_admin=True,
        )
        assert result.success is True
        data = json.loads((workspace.parent / "agent.json").read_text())
        assert data["permissions"] == "open"

    @pytest.mark.asyncio
    async def test_non_admin_cannot_set_open(self, workspace: Path) -> None:
        result = await edit_permissions(
            "open",
            workspace=workspace,
            agent_name="test-agent",
            is_admin=False,
        )
        assert result.success is False
        assert result.error == "insufficient_role"

    @pytest.mark.asyncio
    async def test_unknown_preset_rejected(self, workspace: Path) -> None:
        result = await edit_permissions(
            "nonexistent",
            workspace=workspace,
            agent_name="test-agent",
        )
        assert result.success is False
        assert result.error == "unknown_preset"

    @pytest.mark.asyncio
    async def test_old_profile_logged(self, workspace: Path, db: Database) -> None:
        await edit_permissions(
            "locked",
            workspace=workspace,
            agent_name="test-agent",
            db=db,
        )
        async with db.connection.execute(
            "SELECT detail FROM audit_log WHERE agent_name = ?",
            ("test-agent",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        detail = json.loads(row[0])
        assert detail["old_value"] == "standard"
        assert detail["new_value"] == "locked"

    def test_permission_action_string(self) -> None:
        action = format_action("self_edit", "permissions open")
        profile = get_preset("standard")
        result = check(action, profile)
        from chorus.permissions.engine import PermissionResult

        assert result is PermissionResult.ASK


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TestSelfEditModel:
    @pytest.mark.asyncio
    async def test_updates_agent_json(self, workspace: Path) -> None:
        # No chorus_home → skip validation
        result = await edit_model(
            "gpt-4o",
            workspace=workspace,
            agent_name="test-agent",
        )
        assert result.success is True
        data = json.loads((workspace.parent / "agent.json").read_text())
        assert data["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_validates_against_available(self, workspace: Path, tmp_path: Path) -> None:
        # Write a fake available_models.json cache
        chorus_home = tmp_path / "chorus-home"
        chorus_home.mkdir()
        cache = {
            "providers": {
                "anthropic": {"valid": True, "models": ["claude-sonnet-4-20250514"]},
            },
        }
        (chorus_home / "available_models.json").write_text(json.dumps(cache))

        result = await edit_model(
            "claude-sonnet-4-20250514",
            workspace=workspace,
            agent_name="test-agent",
            chorus_home=chorus_home,
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_rejects_unavailable(self, workspace: Path, tmp_path: Path) -> None:
        chorus_home = tmp_path / "chorus-home"
        chorus_home.mkdir()
        cache = {
            "providers": {
                "anthropic": {"valid": True, "models": ["claude-sonnet-4-20250514"]},
            },
        }
        (chorus_home / "available_models.json").write_text(json.dumps(cache))

        result = await edit_model(
            "nonexistent-model",
            workspace=workspace,
            agent_name="test-agent",
            chorus_home=chorus_home,
        )
        assert result.success is False
        assert result.error == "unavailable_model"

    @pytest.mark.asyncio
    async def test_logs_to_audit(self, workspace: Path, db: Database) -> None:
        await edit_model(
            "gpt-4o",
            workspace=workspace,
            agent_name="test-agent",
            db=db,
        )
        async with db.connection.execute(
            "SELECT action_string, detail FROM audit_log WHERE agent_name = ?",
            ("test-agent",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "tool:self_edit:model"
        detail = json.loads(row[1])
        assert detail["new_value"] == "gpt-4o"

    def test_permission_action_string(self) -> None:
        action = format_action("self_edit", "model gpt-4o")
        profile = get_preset("standard")
        result = check(action, profile)
        from chorus.permissions.engine import PermissionResult

        assert result is PermissionResult.ASK


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


class TestAuditLogging:
    @pytest.mark.asyncio
    async def test_self_edit_logs_to_audit_table(self, db: Database) -> None:
        await db.log_self_edit(
            agent_name="test-agent",
            edit_type="system_prompt",
            old_value="old",
            new_value="new",
        )
        async with db.connection.execute(
            "SELECT agent_name, action_string, decision FROM audit_log",
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "test-agent"
        assert row[1] == "tool:self_edit:system_prompt"
        assert row[2] == "allow"

    @pytest.mark.asyncio
    async def test_audit_log_contains_agent_name(self, db: Database) -> None:
        await db.log_self_edit(
            agent_name="my-agent",
            edit_type="model",
            old_value="old-model",
            new_value="new-model",
        )
        async with db.connection.execute(
            "SELECT agent_name FROM audit_log",
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "my-agent"

    @pytest.mark.asyncio
    async def test_audit_log_contains_old_and_new_values(self, db: Database) -> None:
        await db.log_self_edit(
            agent_name="test-agent",
            edit_type="docs",
            old_value="old content",
            new_value="new content",
        )
        async with db.connection.execute(
            "SELECT detail FROM audit_log",
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        detail = json.loads(row[0])
        assert detail["old_value"] == "old content"
        assert detail["new_value"] == "new content"

    @pytest.mark.asyncio
    async def test_audit_log_truncates_long_values(self, db: Database) -> None:
        long_value = "x" * 1000
        await db.log_self_edit(
            agent_name="test-agent",
            edit_type="system_prompt",
            old_value=long_value,
            new_value="short",
        )
        async with db.connection.execute(
            "SELECT detail FROM audit_log",
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        detail = json.loads(row[0])
        assert len(detail["old_value"]) == 500
        assert detail["new_value"] == "short"

    @pytest.mark.asyncio
    async def test_audit_log_contains_timestamp(self, db: Database) -> None:
        await db.log_self_edit(
            agent_name="test-agent",
            edit_type="permissions",
            old_value="standard",
            new_value="locked",
        )
        async with db.connection.execute(
            "SELECT timestamp FROM audit_log",
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        # ISO format check: should start with year
        assert row[0].startswith("20")
        assert "T" in row[0]


# ---------------------------------------------------------------------------
# Tool integration
# ---------------------------------------------------------------------------


class TestToolIntegration:
    def test_self_edit_tools_registered(self) -> None:
        from chorus.tools.registry import create_default_registry

        registry = create_default_registry()
        expected = [
            "self_edit_system_prompt",
            "self_edit_docs",
            "self_edit_permissions",
            "self_edit_model",
        ]
        for name in expected:
            assert registry.get(name) is not None, f"Tool {name!r} not registered"

    def test_self_edit_tool_schema_correct(self) -> None:
        from chorus.tools.registry import create_default_registry

        registry = create_default_registry()
        # system_prompt
        tool = registry.get("self_edit_system_prompt")
        assert tool is not None
        assert "new_prompt" in tool.parameters["properties"]
        assert "new_prompt" in tool.parameters["required"]

        # docs
        tool = registry.get("self_edit_docs")
        assert tool is not None
        assert "path" in tool.parameters["properties"]
        assert "content" in tool.parameters["properties"]

        # permissions
        tool = registry.get("self_edit_permissions")
        assert tool is not None
        assert "profile" in tool.parameters["properties"]

        # model
        tool = registry.get("self_edit_model")
        assert tool is not None
        assert "model" in tool.parameters["properties"]

    @pytest.mark.asyncio
    async def test_self_edit_tool_invokable_from_tool_loop(
        self, workspace: Path, db: Database
    ) -> None:
        """Verify a self_edit tool can be dispatched through the tool loop's _execute_tool."""
        from chorus.llm.tool_loop import ToolExecutionContext, _execute_tool
        from chorus.tools.registry import create_default_registry

        registry = create_default_registry()
        tool = registry.get("self_edit_system_prompt")
        assert tool is not None

        ctx = ToolExecutionContext(
            workspace=workspace,
            profile=get_preset("open"),
            agent_name="test-agent",
            db=db,
        )
        result_str = await _execute_tool(
            tool,
            {"new_prompt": "Injected prompt."},
            ctx,
        )
        result = json.loads(result_str)
        assert result["success"] is True
        # Verify it actually persisted
        data = json.loads((workspace.parent / "agent.json").read_text())
        assert data["system_prompt"] == "Injected prompt."

    @pytest.mark.asyncio
    async def test_category_mapping(self) -> None:
        """Verify self_edit tools map to the 'self_edit' category for permissions."""
        from chorus.llm.tool_loop import _build_action_string

        action = _build_action_string("self_edit_system_prompt", {"new_prompt": "x"})
        assert action == "tool:self_edit:system_prompt"

        action = _build_action_string("self_edit_docs", {"path": "README.md", "content": "x"})
        assert action == "tool:self_edit:docs README.md"

        action = _build_action_string("self_edit_permissions", {"profile": "open"})
        assert action == "tool:self_edit:permissions open"

        action = _build_action_string("self_edit_model", {"model": "gpt-4o"})
        assert action == "tool:self_edit:model gpt-4o"


# ---------------------------------------------------------------------------
# Fuzzy model name resolution
# ---------------------------------------------------------------------------


class TestResolveShortModelName:
    def test_exact_match_returns_same(self, tmp_path: Path) -> None:
        chorus_home = tmp_path / "chorus-home"
        chorus_home.mkdir()
        cache = {
            "providers": {
                "anthropic": {"valid": True, "models": ["claude-opus-4-20250514"]},
                "openai": {"valid": True, "models": ["gpt-4o"]},
            },
        }
        (chorus_home / "available_models.json").write_text(json.dumps(cache))

        result = _resolve_short_model_name("gpt-4o", chorus_home)
        assert result == "gpt-4o"

    def test_substring_match_resolves(self, tmp_path: Path) -> None:
        chorus_home = tmp_path / "chorus-home"
        chorus_home.mkdir()
        cache = {
            "providers": {
                "anthropic": {
                    "valid": True,
                    "models": ["claude-opus-4-20250514", "claude-sonnet-4-20250514"],
                },
                "openai": {"valid": True, "models": ["gpt-4o"]},
            },
        }
        (chorus_home / "available_models.json").write_text(json.dumps(cache))

        result = _resolve_short_model_name("opus", chorus_home)
        assert result == "claude-opus-4-20250514"

    def test_substring_match_case_insensitive(self, tmp_path: Path) -> None:
        chorus_home = tmp_path / "chorus-home"
        chorus_home.mkdir()
        cache = {
            "providers": {
                "anthropic": {
                    "valid": True,
                    "models": ["claude-sonnet-4-20250514"],
                },
            },
        }
        (chorus_home / "available_models.json").write_text(json.dumps(cache))

        result = _resolve_short_model_name("Sonnet", chorus_home)
        assert result == "claude-sonnet-4-20250514"

    def test_no_match_returns_original(self, tmp_path: Path) -> None:
        chorus_home = tmp_path / "chorus-home"
        chorus_home.mkdir()
        cache = {
            "providers": {
                "anthropic": {"valid": True, "models": ["claude-opus-4-20250514"]},
            },
        }
        (chorus_home / "available_models.json").write_text(json.dumps(cache))

        result = _resolve_short_model_name("nonexistent-model", chorus_home)
        assert result == "nonexistent-model"

    def test_no_chorus_home_returns_original(self) -> None:
        result = _resolve_short_model_name("opus", None)
        assert result == "opus"

    def test_no_cache_file_returns_original(self, tmp_path: Path) -> None:
        chorus_home = tmp_path / "empty-chorus-home"
        chorus_home.mkdir()
        # No available_models.json file
        result = _resolve_short_model_name("opus", chorus_home)
        assert result == "opus"

    @pytest.mark.asyncio
    async def test_edit_model_uses_fuzzy_resolution(
        self, workspace: Path, tmp_path: Path
    ) -> None:
        """Verify edit_model resolves short names before validating."""
        chorus_home = tmp_path / "chorus-home"
        chorus_home.mkdir()
        cache = {
            "providers": {
                "anthropic": {
                    "valid": True,
                    "models": ["claude-opus-4-20250514"],
                },
            },
        }
        (chorus_home / "available_models.json").write_text(json.dumps(cache))

        result = await edit_model(
            "opus",
            workspace=workspace,
            agent_name="test-agent",
            chorus_home=chorus_home,
        )
        assert result.success is True
        assert result.new_value == "claude-opus-4-20250514"
        data = json.loads((workspace.parent / "agent.json").read_text())
        assert data["model"] == "claude-opus-4-20250514"
