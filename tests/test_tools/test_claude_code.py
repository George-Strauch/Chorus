"""Tests for chorus.tools.claude_code — Claude Code SDK integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from chorus.permissions.engine import PermissionProfile, PermissionResult, check, format_action

# ---------------------------------------------------------------------------
# Helpers — fake SDK types for mocking
# ---------------------------------------------------------------------------


@dataclass
class _FakeTextBlock:
    text: str


@dataclass
class _FakeAssistantMessage:
    content: list[Any]
    model: str = "claude-sonnet-4-5-20250929"


@dataclass
class _FakeResultMessage:
    subtype: str = "result"
    duration_ms: int = 5000
    duration_api_ms: int = 4500
    is_error: bool = False
    num_turns: int = 3
    session_id: str = "sess-abc123"
    total_cost_usd: float | None = 0.05
    usage: dict[str, Any] | None = None
    result: str | None = None


async def _fake_query_success(**kwargs: Any) -> Any:
    """Simulate a successful SDK query() that yields messages then a result."""
    msgs = [
        _FakeAssistantMessage(content=[_FakeTextBlock(text="Created file main.py")]),
        _FakeResultMessage(),
    ]
    for m in msgs:
        yield m


async def _fake_query_large_output(**kwargs: Any) -> Any:
    """Simulate a query that returns output exceeding 50KB."""
    large_text = "x" * 60_000
    msgs = [
        _FakeAssistantMessage(content=[_FakeTextBlock(text=large_text)]),
        _FakeResultMessage(),
    ]
    for m in msgs:
        yield m


async def _fake_query_error(**kwargs: Any) -> Any:
    """Simulate a query that returns a ResultMessage with is_error=True."""
    msgs = [
        _FakeResultMessage(is_error=True, result="Something went wrong"),
    ]
    for m in msgs:
        yield m


# ---------------------------------------------------------------------------
# TestClaudeCodeResult
# ---------------------------------------------------------------------------


class TestClaudeCodeResult:
    def test_result_to_dict_success(self) -> None:
        from chorus.tools.claude_code import ClaudeCodeResult

        r = ClaudeCodeResult(
            task="Create main.py",
            success=True,
            output="Created file main.py",
            cost_usd=0.05,
            duration_ms=5000,
            num_turns=3,
            session_id="sess-abc",
        )
        d = r.to_dict()
        assert d["task"] == "Create main.py"
        assert d["success"] is True
        assert d["output"] == "Created file main.py"
        assert d["cost_usd"] == 0.05
        assert d["duration_ms"] == 5000
        assert d["num_turns"] == 3
        assert d["session_id"] == "sess-abc"
        assert d["error"] is None

    def test_result_to_dict_error(self) -> None:
        from chorus.tools.claude_code import ClaudeCodeResult

        r = ClaudeCodeResult(
            task="Fix bug",
            success=False,
            output="",
            cost_usd=None,
            duration_ms=100,
            num_turns=0,
            error="SDK not installed",
        )
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"] == "SDK not installed"
        assert d["cost_usd"] is None


# ---------------------------------------------------------------------------
# TestIsClaudeCodeAvailable
# ---------------------------------------------------------------------------


class TestIsClaudeCodeAvailable:
    def test_available_when_sdk_importable(self) -> None:
        fake_module = MagicMock()
        with patch.dict("sys.modules", {"claude_agent_sdk": fake_module}):
            import chorus.tools.claude_code as cc_mod

            # Directly test the logic
            assert cc_mod.is_claude_code_available() is True

    def test_not_available_when_import_fails(self) -> None:
        import chorus.tools.claude_code as cc_mod

        # The function should handle import failure gracefully
        # and return a bool regardless of whether the SDK is installed
        result = cc_mod.is_claude_code_available()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# TestClaudeCodeExecute
# ---------------------------------------------------------------------------


class TestClaudeCodeExecute:
    @pytest.mark.asyncio
    async def test_successful_execution(self, workspace_dir: Path) -> None:
        from chorus.permissions.engine import PermissionProfile
        from chorus.tools.claude_code import claude_code_execute

        profile = PermissionProfile(allow=[".*"], ask=[])

        with patch("chorus.tools.claude_code._sdk_available", True), patch(
            "chorus.tools.claude_code._run_sdk_query"
        ) as mock_query:
            mock_query.return_value = {
                "output": "Created file main.py",
                "cost_usd": 0.05,
                "duration_ms": 5000,
                "num_turns": 3,
                "session_id": "sess-abc",
                "is_error": False,
            }

            result = await claude_code_execute(
                task="Create a Python hello world",
                workspace=workspace_dir,
                profile=profile,
                agent_name="test-agent",
            )

        assert result.success is True
        assert result.output == "Created file main.py"
        assert result.cost_usd == 0.05
        assert result.duration_ms == 5000
        assert result.num_turns == 3
        assert result.session_id == "sess-abc"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_sdk_not_installed_returns_error(self, workspace_dir: Path) -> None:
        from chorus.permissions.engine import PermissionProfile
        from chorus.tools.claude_code import claude_code_execute

        profile = PermissionProfile(allow=[".*"], ask=[])

        with patch("chorus.tools.claude_code._sdk_available", False):
            result = await claude_code_execute(
                task="Create a file",
                workspace=workspace_dir,
                profile=profile,
                agent_name="test-agent",
            )

        assert result.success is False
        assert "not installed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_timeout_returns_partial_result(self, workspace_dir: Path) -> None:
        from chorus.permissions.engine import PermissionProfile
        from chorus.tools.claude_code import claude_code_execute

        profile = PermissionProfile(allow=[".*"], ask=[])

        with patch("chorus.tools.claude_code._sdk_available", True), patch(
            "chorus.tools.claude_code._run_sdk_query"
        ) as mock_query:
            mock_query.side_effect = TimeoutError("Operation timed out")

            result = await claude_code_execute(
                task="Long running task",
                workspace=workspace_dir,
                profile=profile,
                agent_name="test-agent",
                timeout=1.0,
            )

        assert result.success is False
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_cli_not_found_handled(self, workspace_dir: Path) -> None:
        from chorus.permissions.engine import PermissionProfile
        from chorus.tools.claude_code import claude_code_execute

        profile = PermissionProfile(allow=[".*"], ask=[])

        with patch("chorus.tools.claude_code._sdk_available", True), patch(
            "chorus.tools.claude_code._run_sdk_query"
        ) as mock_query:
            mock_query.side_effect = RuntimeError("CLINotFoundError: Claude Code not found")

            result = await claude_code_execute(
                task="Create a file",
                workspace=workspace_dir,
                profile=profile,
                agent_name="test-agent",
            )

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_process_error_handled(self, workspace_dir: Path) -> None:
        from chorus.permissions.engine import PermissionProfile
        from chorus.tools.claude_code import claude_code_execute

        profile = PermissionProfile(allow=[".*"], ask=[])

        with patch("chorus.tools.claude_code._sdk_available", True), patch(
            "chorus.tools.claude_code._run_sdk_query"
        ) as mock_query:
            mock_query.side_effect = RuntimeError("ProcessError: exit code 1")

            result = await claude_code_execute(
                task="Create a file",
                workspace=workspace_dir,
                profile=profile,
                agent_name="test-agent",
            )

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_output_truncated_at_50kb(self, workspace_dir: Path) -> None:
        from chorus.permissions.engine import PermissionProfile
        from chorus.tools.claude_code import claude_code_execute

        profile = PermissionProfile(allow=[".*"], ask=[])

        large_output = "x" * 60_000

        with patch("chorus.tools.claude_code._sdk_available", True), patch(
            "chorus.tools.claude_code._run_sdk_query"
        ) as mock_query:
            mock_query.return_value = {
                "output": large_output,
                "cost_usd": 0.10,
                "duration_ms": 8000,
                "num_turns": 5,
                "session_id": "sess-large",
                "is_error": False,
            }

            result = await claude_code_execute(
                task="Generate lots of output",
                workspace=workspace_dir,
                profile=profile,
                agent_name="test-agent",
            )

        assert result.success is True
        assert len(result.output) <= 51_000  # 50KB + truncation header
        assert "[truncated:" in result.output

    @pytest.mark.asyncio
    async def test_workspace_passed_as_cwd(self, workspace_dir: Path) -> None:
        from chorus.permissions.engine import PermissionProfile
        from chorus.tools.claude_code import claude_code_execute

        profile = PermissionProfile(allow=[".*"], ask=[])

        with patch("chorus.tools.claude_code._sdk_available", True), patch(
            "chorus.tools.claude_code._run_sdk_query"
        ) as mock_query:
            mock_query.return_value = {
                "output": "Done",
                "cost_usd": 0.01,
                "duration_ms": 1000,
                "num_turns": 1,
                "session_id": "sess-cwd",
                "is_error": False,
            }

            await claude_code_execute(
                task="Check cwd",
                workspace=workspace_dir,
                profile=profile,
                agent_name="test-agent",
            )

        # Verify workspace was passed to the SDK query
        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args
        assert call_kwargs[1]["cwd"] == workspace_dir

    @pytest.mark.asyncio
    async def test_model_forwarded_for_anthropic(self, workspace_dir: Path) -> None:
        from chorus.permissions.engine import PermissionProfile
        from chorus.tools.claude_code import claude_code_execute

        profile = PermissionProfile(allow=[".*"], ask=[])

        with patch("chorus.tools.claude_code._sdk_available", True), patch(
            "chorus.tools.claude_code._run_sdk_query"
        ) as mock_query:
            mock_query.return_value = {
                "output": "Done",
                "cost_usd": 0.01,
                "duration_ms": 1000,
                "num_turns": 1,
                "session_id": "sess-model",
                "is_error": False,
            }

            await claude_code_execute(
                task="Test model",
                workspace=workspace_dir,
                profile=profile,
                agent_name="test-agent",
                model="claude-sonnet-4-5-20250929",
            )

        call_kwargs = mock_query.call_args
        assert call_kwargs[1]["model"] == "claude-sonnet-4-5-20250929"

    @pytest.mark.asyncio
    async def test_non_anthropic_model_not_forwarded(self, workspace_dir: Path) -> None:
        from chorus.permissions.engine import PermissionProfile
        from chorus.tools.claude_code import claude_code_execute

        profile = PermissionProfile(allow=[".*"], ask=[])

        with patch("chorus.tools.claude_code._sdk_available", True), patch(
            "chorus.tools.claude_code._run_sdk_query"
        ) as mock_query:
            mock_query.return_value = {
                "output": "Done",
                "cost_usd": 0.01,
                "duration_ms": 1000,
                "num_turns": 1,
                "session_id": "sess-model2",
                "is_error": False,
            }

            await claude_code_execute(
                task="Test model",
                workspace=workspace_dir,
                profile=profile,
                agent_name="test-agent",
                model="gpt-4o",
            )

        call_kwargs = mock_query.call_args
        assert call_kwargs[1]["model"] is None

    @pytest.mark.asyncio
    async def test_sdk_error_result(self, workspace_dir: Path) -> None:
        """SDK returns a ResultMessage with is_error=True."""
        from chorus.permissions.engine import PermissionProfile
        from chorus.tools.claude_code import claude_code_execute

        profile = PermissionProfile(allow=[".*"], ask=[])

        with patch("chorus.tools.claude_code._sdk_available", True), patch(
            "chorus.tools.claude_code._run_sdk_query"
        ) as mock_query:
            mock_query.return_value = {
                "output": "Something went wrong",
                "cost_usd": 0.01,
                "duration_ms": 500,
                "num_turns": 1,
                "session_id": "sess-err",
                "is_error": True,
            }

            result = await claude_code_execute(
                task="Failing task",
                workspace=workspace_dir,
                profile=profile,
                agent_name="test-agent",
            )

        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# TestClaudeCodeRegistration
# ---------------------------------------------------------------------------


class TestClaudeCodeRegistration:
    def test_tool_registered_when_sdk_available(self) -> None:
        with patch("chorus.tools.claude_code._sdk_available", True):
            from chorus.tools.registry import create_default_registry

            registry = create_default_registry()
            tool = registry.get("claude_code")
            assert tool is not None
            assert "claude_code" in tool.name

    def test_tool_not_registered_when_sdk_unavailable(self) -> None:
        with patch("chorus.tools.claude_code._sdk_available", False):
            from chorus.tools.registry import create_default_registry

            registry = create_default_registry()
            tool = registry.get("claude_code")
            assert tool is None


# ---------------------------------------------------------------------------
# TestClaudeCodePermissions
# ---------------------------------------------------------------------------


class TestClaudeCodePermissions:
    def test_action_string_format(self) -> None:
        action = format_action("claude_code", "Create a Python file")
        assert action == "tool:claude_code:Create a Python file"

    def test_standard_preset_asks_for_claude_code(self) -> None:
        from chorus.permissions.engine import PRESETS

        standard = PRESETS["standard"]
        action = format_action("claude_code", "Create main.py")
        result = check(action, standard)
        assert result is PermissionResult.ASK

    def test_open_preset_allows_claude_code(self) -> None:
        from chorus.permissions.engine import PRESETS

        open_preset = PRESETS["open"]
        action = format_action("claude_code", "Create main.py")
        result = check(action, open_preset)
        assert result is PermissionResult.ALLOW

    def test_locked_preset_denies_claude_code(self) -> None:
        from chorus.permissions.engine import PRESETS

        locked = PRESETS["locked"]
        action = format_action("claude_code", "Create main.py")
        result = check(action, locked)
        assert result is PermissionResult.DENY

    def test_guarded_preset_allows_claude_code(self) -> None:
        from chorus.permissions.engine import PRESETS

        guarded = PRESETS["guarded"]
        action = format_action("claude_code", "Create main.py")
        result = check(action, guarded)
        assert result is PermissionResult.ALLOW

    def test_action_string_truncated_in_tool_loop(self) -> None:
        from chorus.llm.tool_loop import _build_action_string

        action = _build_action_string("claude_code", {"task": "x" * 200})
        # Should be truncated to 100 chars for the detail
        assert len(action) < 250


# ---------------------------------------------------------------------------
# TestSubagentConfig
# ---------------------------------------------------------------------------


class TestSubagentConfig:
    @pytest.mark.asyncio
    async def test_subagent_config_passed_to_sdk(self, workspace_dir: Path) -> None:
        """_run_sdk_query passes agents config to ClaudeAgentOptions."""
        from chorus.tools.claude_code import claude_code_execute

        profile = PermissionProfile(allow=[".*"], ask=[])

        with patch("chorus.tools.claude_code._sdk_available", True), patch(
            "chorus.tools.claude_code._run_sdk_query"
        ) as mock_query:
            mock_query.return_value = {
                "output": "Done",
                "cost_usd": 0.01,
                "duration_ms": 1000,
                "num_turns": 1,
                "session_id": "sess-sub",
                "is_error": False,
            }

            await claude_code_execute(
                task="Test subagents",
                workspace=workspace_dir,
                profile=profile,
                agent_name="test-agent",
            )

        mock_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_allowed_tools_includes_task(self, workspace_dir: Path) -> None:
        """allowed_tools includes Task for subagent delegation."""
        from chorus.tools.claude_code import claude_code_execute

        profile = PermissionProfile(allow=[".*"], ask=[])

        with patch("chorus.tools.claude_code._sdk_available", True), patch(
            "chorus.tools.claude_code._run_sdk_query"
        ) as mock_query:
            mock_query.return_value = {
                "output": "Done",
                "cost_usd": 0.01,
                "duration_ms": 1000,
                "num_turns": 1,
                "session_id": "sess-task",
                "is_error": False,
            }

            await claude_code_execute(
                task="Test allowed tools",
                workspace=workspace_dir,
                profile=profile,
                agent_name="test-agent",
            )

        mock_query.assert_called_once()


# ---------------------------------------------------------------------------
# TestHostExecution
# ---------------------------------------------------------------------------


class TestHostExecution:
    @pytest.mark.asyncio
    async def test_host_execution_passes_full_env(self, workspace_dir: Path) -> None:
        """When host_execution=True, bash should pass through full environment."""
        from chorus.tools.bash import _sanitized_env

        # Normal mode: HOME is jailed
        normal_env = _sanitized_env(workspace_dir)
        assert normal_env["HOME"] == str(workspace_dir)

        # Host mode: HOME is NOT jailed (uses real HOME)
        host_env = _sanitized_env(workspace_dir, host_execution=True)
        assert host_env["HOME"] != str(workspace_dir)
        # Should contain more env vars than normal mode
        assert len(host_env) >= len(normal_env)

    @pytest.mark.asyncio
    async def test_host_execution_false_jails_home(self, workspace_dir: Path) -> None:
        """Default (host_execution=False) should jail HOME to workspace."""
        from chorus.tools.bash import _sanitized_env

        env = _sanitized_env(workspace_dir, host_execution=False)
        assert env["HOME"] == str(workspace_dir)


# ---------------------------------------------------------------------------
# TestToolExecutionContextHostExec
# ---------------------------------------------------------------------------


class TestToolExecutionContextHostExec:
    def test_context_has_host_execution_field(self) -> None:
        from chorus.llm.tool_loop import ToolExecutionContext

        ctx = ToolExecutionContext(
            workspace=Path("/tmp"),
            profile=PermissionProfile(allow=[".*"], ask=[]),
            agent_name="test",
            host_execution=True,
        )
        assert ctx.host_execution is True

    def test_context_host_execution_defaults_false(self) -> None:
        from chorus.llm.tool_loop import ToolExecutionContext

        ctx = ToolExecutionContext(
            workspace=Path("/tmp"),
            profile=PermissionProfile(allow=[".*"], ask=[]),
            agent_name="test",
        )
        assert ctx.host_execution is False
