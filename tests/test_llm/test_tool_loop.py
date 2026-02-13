"""Tests for chorus.llm.tool_loop — agentic tool use loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chorus.llm.providers import LLMResponse, ToolCall, Usage
from chorus.llm.tool_loop import (
    ToolExecutionContext,
    ToolLoopResult,
    run_tool_loop,
)
from chorus.permissions.engine import PermissionProfile
from chorus.tools.registry import ToolDefinition, ToolRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_response(
    text: str = "Done.",
    model: str = "claude-sonnet-4-20250514",
) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=[],
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=5),
        model=model,
    )


def _tool_response(
    tool_calls: list[ToolCall],
    text: str | None = "Calling tools...",
    model: str = "claude-sonnet-4-20250514",
) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=tool_calls,
        stop_reason="tool_use",
        usage=Usage(input_tokens=20, output_tokens=15),
        model=model,
    )


class FakeProvider:
    """A fake LLM provider that returns scripted responses in order."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.call_log: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        self.call_log.append({"messages": messages, "tools": tools, "model": model})
        if self._call_count >= len(self._responses):
            return _text_response("(ran out of scripted responses)")
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp


def _make_registry(*tool_defs: tuple[str, AsyncMock]) -> ToolRegistry:
    """Build a registry with simple tool definitions."""
    registry = ToolRegistry()
    for name, handler in tool_defs:
        registry.register(
            ToolDefinition(
                name=name,
                description=f"Tool {name}",
                parameters={
                    "type": "object",
                    "properties": {
                        "arg": {"type": "string", "description": "An argument"},
                    },
                    "required": ["arg"],
                },
                handler=handler,
            )
        )
    return registry


def _open_profile() -> PermissionProfile:
    return PermissionProfile(allow=[".*"], ask=[])


def _deny_profile() -> PermissionProfile:
    return PermissionProfile(allow=[], ask=[])


def _ask_profile() -> PermissionProfile:
    return PermissionProfile(allow=[], ask=[".*"])


def _make_ctx(tmp_path: Path) -> ToolExecutionContext:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return ToolExecutionContext(
        workspace=workspace,
        profile=_open_profile(),
        agent_name="test-agent",
    )


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


class TestToolLoopCore:
    @pytest.mark.asyncio
    async def test_returns_text_response_immediately(self, tmp_path: Path) -> None:
        """No tool calls → returns immediately."""
        provider = FakeProvider([_text_response("Hello!")])
        ctx = _make_ctx(tmp_path)
        registry = _make_registry()

        result = await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Hi"}],
            tools=registry,
            ctx=ctx,
            system_prompt="You are helpful.",
            model="claude-sonnet-4-20250514",
        )

        assert isinstance(result, ToolLoopResult)
        assert result.content == "Hello!"
        assert result.iterations == 1
        assert result.tool_calls_made == 0

    @pytest.mark.asyncio
    async def test_executes_single_tool_call(self, tmp_path: Path) -> None:
        handler = AsyncMock(return_value={"result": "file created"})
        registry = _make_registry(("create_file", handler))

        provider = FakeProvider([
            _tool_response([ToolCall(id="tc_1", name="create_file", arguments={"arg": "test"})]),
            _text_response("File created successfully."),
        ])

        ctx = _make_ctx(tmp_path)
        result = await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Create a file"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="claude-sonnet-4-20250514",
        )

        assert result.content == "File created successfully."
        assert result.iterations == 2
        assert result.tool_calls_made == 1
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_executes_multiple_tool_calls_in_sequence(self, tmp_path: Path) -> None:
        h1 = AsyncMock(return_value={"result": "done 1"})
        h2 = AsyncMock(return_value={"result": "done 2"})
        registry = _make_registry(("tool_a", h1), ("tool_b", h2))

        provider = FakeProvider([
            _tool_response([
                ToolCall(id="tc_1", name="tool_a", arguments={"arg": "x"}),
                ToolCall(id="tc_2", name="tool_b", arguments={"arg": "y"}),
            ]),
            _text_response("Both done."),
        ])

        ctx = _make_ctx(tmp_path)
        result = await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Do both"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="claude-sonnet-4-20250514",
        )

        assert result.content == "Both done."
        assert result.tool_calls_made == 2
        h1.assert_called_once()
        h2.assert_called_once()

    @pytest.mark.asyncio
    async def test_feeds_tool_result_back_to_llm(self, tmp_path: Path) -> None:
        handler = AsyncMock(return_value={"output": "42"})
        registry = _make_registry(("compute", handler))

        provider = FakeProvider([
            _tool_response([ToolCall(id="tc_1", name="compute", arguments={"arg": "6*7"})]),
            _text_response("The answer is 42."),
        ])

        ctx = _make_ctx(tmp_path)
        await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Compute 6*7"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="claude-sonnet-4-20250514",
        )

        # The second call should include the tool result message
        second_call_msgs = provider.call_log[1]["messages"]
        tool_result_msgs = [m for m in second_call_msgs if m.get("role") == "tool"]
        assert len(tool_result_msgs) == 1
        assert "42" in tool_result_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_multi_turn_tool_conversation(self, tmp_path: Path) -> None:
        handler = AsyncMock(side_effect=[{"step": "1"}, {"step": "2"}])
        registry = _make_registry(("step_tool", handler))

        provider = FakeProvider([
            _tool_response([ToolCall(id="tc_1", name="step_tool", arguments={"arg": "1"})]),
            _tool_response([ToolCall(id="tc_2", name="step_tool", arguments={"arg": "2"})]),
            _text_response("Both steps complete."),
        ])

        ctx = _make_ctx(tmp_path)
        result = await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Run steps"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="claude-sonnet-4-20250514",
        )

        assert result.content == "Both steps complete."
        assert result.iterations == 3
        assert result.tool_calls_made == 2

    @pytest.mark.asyncio
    async def test_max_iterations_stops_infinite_loop(self, tmp_path: Path) -> None:
        handler = AsyncMock(return_value={"ok": True})
        registry = _make_registry(("looper", handler))

        # Provider always returns tool calls — loop should cap at max_iterations
        responses = [
            _tool_response([ToolCall(id=f"tc_{i}", name="looper", arguments={"arg": str(i)})])
            for i in range(100)
        ]
        provider = FakeProvider(responses)

        ctx = _make_ctx(tmp_path)
        result = await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Loop forever"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="claude-sonnet-4-20250514",
            max_iterations=3,
        )

        assert result.iterations == 3
        assert "max iterations" in (result.content or "").lower()


# ---------------------------------------------------------------------------
# Permission integration
# ---------------------------------------------------------------------------


class TestToolLoopPermissions:
    @pytest.mark.asyncio
    async def test_allowed_tool_executes_automatically(self, tmp_path: Path) -> None:
        handler = AsyncMock(return_value={"ok": True})
        registry = _make_registry(("my_tool", handler))

        provider = FakeProvider([
            _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
            _text_response("Done."),
        ])

        ctx = _make_ctx(tmp_path)
        ctx.profile = _open_profile()

        result = await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Do it"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
        )

        handler.assert_called_once()
        assert result.content == "Done."

    @pytest.mark.asyncio
    async def test_denied_tool_returns_error(self, tmp_path: Path) -> None:
        handler = AsyncMock(return_value={"ok": True})
        registry = _make_registry(("my_tool", handler))

        provider = FakeProvider([
            _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
            _text_response("Permission denied, sorry."),
        ])

        ctx = _make_ctx(tmp_path)
        ctx.profile = _deny_profile()

        await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Do it"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
        )

        handler.assert_not_called()
        # The error should have been fed back to the LLM
        second_call_msgs = provider.call_log[1]["messages"]
        tool_results = [m for m in second_call_msgs if m.get("role") == "tool"]
        assert any("denied" in (m.get("content", "")).lower() for m in tool_results)

    @pytest.mark.asyncio
    async def test_ask_tool_prompts_callback_and_approved(self, tmp_path: Path) -> None:
        handler = AsyncMock(return_value={"ok": True})
        registry = _make_registry(("my_tool", handler))

        provider = FakeProvider([
            _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
            _text_response("Approved and done."),
        ])

        ctx = _make_ctx(tmp_path)
        ctx.profile = _ask_profile()

        ask_callback = AsyncMock(return_value=True)

        result = await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Do it"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
            ask_callback=ask_callback,
        )

        ask_callback.assert_called_once()
        handler.assert_called_once()
        assert result.content == "Approved and done."

    @pytest.mark.asyncio
    async def test_ask_tool_prompts_callback_and_rejected(self, tmp_path: Path) -> None:
        handler = AsyncMock(return_value={"ok": True})
        registry = _make_registry(("my_tool", handler))

        provider = FakeProvider([
            _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
            _text_response("User declined."),
        ])

        ctx = _make_ctx(tmp_path)
        ctx.profile = _ask_profile()

        ask_callback = AsyncMock(return_value=False)

        await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Do it"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
            ask_callback=ask_callback,
        )

        ask_callback.assert_called_once()
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_ask_tool_no_callback_defaults_to_deny(self, tmp_path: Path) -> None:
        handler = AsyncMock(return_value={"ok": True})
        registry = _make_registry(("my_tool", handler))

        provider = FakeProvider([
            _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
            _text_response("No callback available."),
        ])

        ctx = _make_ctx(tmp_path)
        ctx.profile = _ask_profile()

        await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Do it"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
            ask_callback=None,
        )

        handler.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestToolLoopErrors:
    @pytest.mark.asyncio
    async def test_tool_execution_error_fed_back(self, tmp_path: Path) -> None:
        handler = AsyncMock(side_effect=RuntimeError("disk full"))
        registry = _make_registry(("failing_tool", handler))

        provider = FakeProvider([
            _tool_response([ToolCall(id="tc_1", name="failing_tool", arguments={"arg": "x"})]),
            _text_response("Tool failed, I'll try something else."),
        ])

        ctx = _make_ctx(tmp_path)
        await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Do it"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
        )

        # Error should have been sent back to the LLM
        second_call_msgs = provider.call_log[1]["messages"]
        tool_results = [m for m in second_call_msgs if m.get("role") == "tool"]
        assert any("disk full" in (m.get("content", "")).lower() for m in tool_results)

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, tmp_path: Path) -> None:
        registry = _make_registry()  # empty registry

        provider = FakeProvider([
            _tool_response([ToolCall(id="tc_1", name="nonexistent", arguments={"arg": "x"})]),
            _text_response("Tool not found, sorry."),
        ])

        ctx = _make_ctx(tmp_path)
        await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Do it"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
        )

        # Error message should mention the unknown tool
        second_call_msgs = provider.call_log[1]["messages"]
        tool_results = [m for m in second_call_msgs if m.get("role") == "tool"]
        assert any("unknown tool" in (m.get("content", "")).lower() for m in tool_results)

    @pytest.mark.asyncio
    async def test_llm_api_error_raised(self, tmp_path: Path) -> None:
        """Errors from the LLM API should propagate up."""

        class ErrorProvider:
            @property
            def provider_name(self) -> str:
                return "error"

            async def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                model: str | None = None,
            ) -> LLMResponse:
                raise ConnectionError("API unreachable")

        ctx = _make_ctx(tmp_path)
        with pytest.raises(ConnectionError, match="API unreachable"):
            await run_tool_loop(
                provider=ErrorProvider(),
                messages=[{"role": "user", "content": "Hi"}],
                tools=_make_registry(),
                ctx=ctx,
                system_prompt="",
                model="test",
            )


# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------


class TestToolLoopUsage:
    @pytest.mark.asyncio
    async def test_tracks_total_token_usage(self, tmp_path: Path) -> None:
        handler = AsyncMock(return_value={"ok": True})
        registry = _make_registry(("my_tool", handler))

        provider = FakeProvider([
            _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
            _text_response("Done."),
        ])

        ctx = _make_ctx(tmp_path)
        result = await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Do it"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
        )

        # First call: 20 in + 15 out, second call: 10 in + 5 out
        assert result.total_usage.input_tokens == 30
        assert result.total_usage.output_tokens == 20

    @pytest.mark.asyncio
    async def test_usage_on_single_turn(self, tmp_path: Path) -> None:
        provider = FakeProvider([_text_response("Hi")])
        ctx = _make_ctx(tmp_path)

        result = await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Hi"}],
            tools=_make_registry(),
            ctx=ctx,
            system_prompt="",
            model="test",
        )

        assert result.total_usage.input_tokens == 10
        assert result.total_usage.output_tokens == 5


# ---------------------------------------------------------------------------
# Context injection (workspace, profile, agent_name)
# ---------------------------------------------------------------------------


class TestToolContextInjection:
    @pytest.mark.asyncio
    async def test_workspace_injected_into_tool(self, tmp_path: Path) -> None:
        """Tools that accept workspace should get it injected from ctx."""
        received: dict[str, Any] = {}

        async def fake_create_file(workspace: Path, path: str, content: str) -> dict[str, Any]:
            received["workspace"] = workspace
            received["path"] = path
            received["content"] = content
            return {"ok": True}

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="create_file",
                description="Create a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
                handler=fake_create_file,
            )
        )

        provider = FakeProvider([
            _tool_response([
                ToolCall(
                    id="tc_1",
                    name="create_file",
                    arguments={"path": "test.txt", "content": "hello"},
                )
            ]),
            _text_response("Done."),
        ])

        ctx = _make_ctx(tmp_path)
        await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Create file"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
        )

        # Handler should have been called with workspace injected
        assert "workspace" in received
        assert isinstance(received["workspace"], Path)
        assert received["path"] == "test.txt"
        assert received["content"] == "hello"
