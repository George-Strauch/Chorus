"""Tests for chorus.tools.run_process — run_concurrent and run_background."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chorus.permissions.engine import PermissionProfile
from chorus.process.models import ProcessCallback, ProcessType, TrackedProcess
from chorus.tools.run_process import run_background, run_concurrent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOW_ALL = PermissionProfile(allow=[".*"], ask=[])


def _make_tracked(pid: int = 42, **kwargs: Any) -> TrackedProcess:
    """Build a minimal TrackedProcess for test purposes."""
    from collections import deque
    from pathlib import Path

    defaults: dict[str, Any] = {
        "pid": pid,
        "command": "echo test",
        "working_directory": Path("/tmp/test"),
        "agent_name": "test-agent",
        "started_at": "2026-01-01T00:00:00+00:00",
        "process_type": ProcessType.CONCURRENT,
        "status": MagicMock(value="running"),
        "callbacks": [],
        "rolling_tail": deque(),
    }
    defaults.update(kwargs)
    return MagicMock(spec=TrackedProcess, **defaults)


@pytest.fixture
def mock_process_manager() -> AsyncMock:
    pm = AsyncMock()
    tracked = _make_tracked(pid=42, callbacks=[])
    pm.spawn.return_value = tracked
    return pm


@pytest.fixture
def workspace_dir(tmp_path: Any) -> Any:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


# ---------------------------------------------------------------------------
# run_concurrent
# ---------------------------------------------------------------------------


class TestRunConcurrent:
    @pytest.mark.asyncio
    async def test_returns_pid_and_status(
        self, workspace_dir: Any, mock_process_manager: AsyncMock
    ) -> None:
        with patch(
            "chorus.tools.run_process.build_callbacks_from_instructions",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await run_concurrent(
                command="echo hello",
                instructions="",
                workspace=workspace_dir,
                profile=ALLOW_ALL,
                agent_name="test-agent",
                process_manager=mock_process_manager,
            )

        assert result["pid"] == 42
        assert result["status"] == "running"
        assert result["type"] == "concurrent"
        assert "message" in result

    @pytest.mark.asyncio
    async def test_passes_branch_id_to_spawn(
        self, workspace_dir: Any, mock_process_manager: AsyncMock
    ) -> None:
        with patch(
            "chorus.tools.run_process.build_callbacks_from_instructions",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await run_concurrent(
                command="echo hello",
                instructions="",
                workspace=workspace_dir,
                profile=ALLOW_ALL,
                agent_name="test-agent",
                process_manager=mock_process_manager,
                branch_id=7,
            )

        mock_process_manager.spawn.assert_called_once()
        call_kwargs = mock_process_manager.spawn.call_args[1]
        assert call_kwargs["spawned_by_branch"] == 7

    @pytest.mark.asyncio
    async def test_blocked_command_returns_error(
        self, workspace_dir: Any, mock_process_manager: AsyncMock
    ) -> None:
        result = await run_concurrent(
            command="rm -rf /",
            instructions="",
            workspace=workspace_dir,
            profile=ALLOW_ALL,
            agent_name="test-agent",
            process_manager=mock_process_manager,
        )

        assert "error" in result
        mock_process_manager.spawn.assert_not_called()

    @pytest.mark.asyncio
    async def test_spawns_concurrent_type(
        self, workspace_dir: Any, mock_process_manager: AsyncMock
    ) -> None:
        with patch(
            "chorus.tools.run_process.build_callbacks_from_instructions",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await run_concurrent(
                command="echo hello",
                instructions="",
                workspace=workspace_dir,
                profile=ALLOW_ALL,
                agent_name="test-agent",
                process_manager=mock_process_manager,
            )

        call_kwargs = mock_process_manager.spawn.call_args[1]
        assert call_kwargs["process_type"] == ProcessType.CONCURRENT

    @pytest.mark.asyncio
    async def test_callbacks_included_in_result(
        self, workspace_dir: Any, mock_process_manager: AsyncMock
    ) -> None:
        mock_cb = MagicMock(spec=ProcessCallback)
        mock_cb.to_dict.return_value = {"trigger": "on_exit", "action": "spawn_branch"}
        tracked = _make_tracked(pid=42, callbacks=[mock_cb])
        mock_process_manager.spawn.return_value = tracked

        with patch(
            "chorus.tools.run_process.build_callbacks_from_instructions",
            new_callable=AsyncMock,
            return_value=[mock_cb],
        ):
            result = await run_concurrent(
                command="echo hello",
                instructions="notify on exit",
                workspace=workspace_dir,
                profile=ALLOW_ALL,
                agent_name="test-agent",
                process_manager=mock_process_manager,
            )

        assert len(result["callbacks"]) == 1
        assert result["callbacks"][0]["action"] == "spawn_branch"


# ---------------------------------------------------------------------------
# run_background
# ---------------------------------------------------------------------------


class TestRunBackground:
    @pytest.mark.asyncio
    async def test_returns_pid_and_status(
        self, workspace_dir: Any, mock_process_manager: AsyncMock
    ) -> None:
        tracked = _make_tracked(pid=99, process_type=ProcessType.BACKGROUND, callbacks=[])
        mock_process_manager.spawn.return_value = tracked

        with patch(
            "chorus.tools.run_process.build_callbacks_from_instructions",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await run_background(
                command="python server.py",
                instructions="",
                workspace=workspace_dir,
                profile=ALLOW_ALL,
                agent_name="test-agent",
                process_manager=mock_process_manager,
            )

        assert result["pid"] == 99
        assert result["status"] == "running"
        assert result["type"] == "background"

    @pytest.mark.asyncio
    async def test_spawns_background_type(
        self, workspace_dir: Any, mock_process_manager: AsyncMock
    ) -> None:
        with patch(
            "chorus.tools.run_process.build_callbacks_from_instructions",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await run_background(
                command="python server.py",
                instructions="",
                workspace=workspace_dir,
                profile=ALLOW_ALL,
                agent_name="test-agent",
                process_manager=mock_process_manager,
            )

        call_kwargs = mock_process_manager.spawn.call_args[1]
        assert call_kwargs["process_type"] == ProcessType.BACKGROUND

    @pytest.mark.asyncio
    async def test_passes_model_for_hooks(
        self, workspace_dir: Any, mock_process_manager: AsyncMock
    ) -> None:
        with patch(
            "chorus.tools.run_process.build_callbacks_from_instructions",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await run_background(
                command="python server.py",
                instructions="",
                model="claude-haiku-4-5-20251001",
                workspace=workspace_dir,
                profile=ALLOW_ALL,
                agent_name="test-agent",
                process_manager=mock_process_manager,
            )

        call_kwargs = mock_process_manager.spawn.call_args[1]
        assert call_kwargs["model_for_hooks"] == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_blocked_command_returns_error(
        self, workspace_dir: Any, mock_process_manager: AsyncMock
    ) -> None:
        result = await run_background(
            command=":(){ :|:& };:",
            instructions="",
            workspace=workspace_dir,
            profile=ALLOW_ALL,
            agent_name="test-agent",
            process_manager=mock_process_manager,
        )

        assert "error" in result
        mock_process_manager.spawn.assert_not_called()

    @pytest.mark.asyncio
    async def test_working_directory_used(
        self, workspace_dir: Any, mock_process_manager: AsyncMock
    ) -> None:
        """working_directory param overrides workspace for spawn."""
        subdir = workspace_dir / "subproject"
        subdir.mkdir()

        with patch(
            "chorus.tools.run_process.build_callbacks_from_instructions",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await run_background(
                command="make build",
                instructions="",
                working_directory=str(subdir),
                workspace=workspace_dir,
                profile=ALLOW_ALL,
                agent_name="test-agent",
                process_manager=mock_process_manager,
            )

        call_kwargs = mock_process_manager.spawn.call_args[1]
        assert str(call_kwargs["workspace"]) == str(subdir.resolve())

    @pytest.mark.asyncio
    async def test_working_directory_relative(
        self, workspace_dir: Any, mock_process_manager: AsyncMock
    ) -> None:
        """Relative working_directory is resolved against workspace."""
        subdir = workspace_dir / "src"
        subdir.mkdir()

        with patch(
            "chorus.tools.run_process.build_callbacks_from_instructions",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await run_concurrent(
                command="echo test",
                working_directory="src",
                workspace=workspace_dir,
                profile=ALLOW_ALL,
                agent_name="test-agent",
                process_manager=mock_process_manager,
            )

        call_kwargs = mock_process_manager.spawn.call_args[1]
        assert str(call_kwargs["workspace"]) == str(subdir.resolve())

    @pytest.mark.asyncio
    async def test_working_directory_traversal_blocked(
        self, workspace_dir: Any, mock_process_manager: AsyncMock
    ) -> None:
        """Path traversal in working_directory is blocked."""
        result = await run_concurrent(
            command="echo test",
            working_directory="/etc",
            workspace=workspace_dir,
            profile=ALLOW_ALL,
            agent_name="test-agent",
            process_manager=mock_process_manager,
        )

        assert "error" in result
        mock_process_manager.spawn.assert_not_called()

    @pytest.mark.asyncio
    async def test_instructions_passed_as_context(
        self, workspace_dir: Any, mock_process_manager: AsyncMock
    ) -> None:
        with patch(
            "chorus.tools.run_process.build_callbacks_from_instructions",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await run_background(
                command="make build",
                instructions="notify me when it finishes",
                workspace=workspace_dir,
                profile=ALLOW_ALL,
                agent_name="test-agent",
                process_manager=mock_process_manager,
            )

        call_kwargs = mock_process_manager.spawn.call_args[1]
        assert call_kwargs["context"] == "notify me when it finishes"


# ---------------------------------------------------------------------------
# Integration: _TOOL_TO_CATEGORY mapping
# ---------------------------------------------------------------------------


class TestToolCategoryMapping:
    def test_run_concurrent_mapped(self) -> None:
        from chorus.llm.tool_loop import _TOOL_TO_CATEGORY

        assert _TOOL_TO_CATEGORY["run_concurrent"] == "run_concurrent"

    def test_run_background_mapped(self) -> None:
        from chorus.llm.tool_loop import _TOOL_TO_CATEGORY

        assert _TOOL_TO_CATEGORY["run_background"] == "run_background"

    def test_action_string_uses_command(self) -> None:
        from chorus.llm.tool_loop import _build_action_string

        action = _build_action_string("run_concurrent", {"command": "echo hi"})
        assert action == "tool:run_concurrent:echo hi"

    def test_action_string_background(self) -> None:
        from chorus.llm.tool_loop import _build_action_string

        action = _build_action_string("run_background", {"command": "python server.py"})
        assert action == "tool:run_background:python server.py"


# ---------------------------------------------------------------------------
# Integration: ToolExecutionContext has new fields
# ---------------------------------------------------------------------------


class TestToolExecutionContextExtended:
    def test_context_has_process_manager(self) -> None:
        from pathlib import Path

        from chorus.llm.tool_loop import ToolExecutionContext

        ctx = ToolExecutionContext(
            workspace=Path("/tmp"),
            profile=ALLOW_ALL,
            agent_name="test",
            process_manager="pm_mock",
        )
        assert ctx.process_manager == "pm_mock"

    def test_context_has_branch_id(self) -> None:
        from pathlib import Path

        from chorus.llm.tool_loop import ToolExecutionContext

        ctx = ToolExecutionContext(
            workspace=Path("/tmp"),
            profile=ALLOW_ALL,
            agent_name="test",
            branch_id=5,
        )
        assert ctx.branch_id == 5

    def test_context_defaults_to_none(self) -> None:
        from pathlib import Path

        from chorus.llm.tool_loop import ToolExecutionContext

        ctx = ToolExecutionContext(
            workspace=Path("/tmp"),
            profile=ALLOW_ALL,
            agent_name="test",
        )
        assert ctx.process_manager is None
        assert ctx.branch_id is None


# ---------------------------------------------------------------------------
# Integration: permission patterns
# ---------------------------------------------------------------------------


class TestPermissionPatterns:
    def test_run_concurrent_ask_in_standard(self) -> None:
        from chorus.permissions.engine import PRESETS, PermissionResult, check

        profile = PRESETS["standard"]
        result = check("tool:run_concurrent:echo hello", profile)
        assert result == PermissionResult.ASK

    def test_run_background_ask_in_standard(self) -> None:
        from chorus.permissions.engine import PRESETS, PermissionResult, check

        profile = PRESETS["standard"]
        result = check("tool:run_background:python server.py", profile)
        assert result == PermissionResult.ASK

    def test_run_concurrent_allow_in_open(self) -> None:
        from chorus.permissions.engine import PRESETS, PermissionResult, check

        profile = PRESETS["open"]
        result = check("tool:run_concurrent:echo hello", profile)
        assert result == PermissionResult.ALLOW

    def test_run_concurrent_deny_in_locked(self) -> None:
        from chorus.permissions.engine import PRESETS, PermissionResult, check

        profile = PRESETS["locked"]
        result = check("tool:run_concurrent:echo hello", profile)
        assert result == PermissionResult.DENY


# ---------------------------------------------------------------------------
# Registry registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_run_concurrent_registered(self) -> None:
        from chorus.tools.registry import create_default_registry

        registry = create_default_registry()
        tool = registry.get("run_concurrent")
        assert tool is not None
        assert tool.name == "run_concurrent"

    def test_run_background_registered(self) -> None:
        from chorus.tools.registry import create_default_registry

        registry = create_default_registry()
        tool = registry.get("run_background")
        assert tool is not None
        assert tool.name == "run_background"
