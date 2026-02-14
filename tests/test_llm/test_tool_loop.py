"""Tests for chorus.llm.tool_loop — agentic tool use loop."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chorus.llm.providers import LLMResponse, ToolCall, Usage
from chorus.llm.tool_loop import (
    ToolExecutionContext,
    ToolLoopEvent,
    ToolLoopEventType,
    ToolLoopResult,
    _truncate_tool_loop_messages,
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

        provider = FakeProvider(
            [
                _tool_response(
                    [ToolCall(id="tc_1", name="create_file", arguments={"arg": "test"})]
                ),
                _text_response("File created successfully."),
            ]
        )

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

        provider = FakeProvider(
            [
                _tool_response(
                    [
                        ToolCall(id="tc_1", name="tool_a", arguments={"arg": "x"}),
                        ToolCall(id="tc_2", name="tool_b", arguments={"arg": "y"}),
                    ]
                ),
                _text_response("Both done."),
            ]
        )

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

        provider = FakeProvider(
            [
                _tool_response([ToolCall(id="tc_1", name="compute", arguments={"arg": "6*7"})]),
                _text_response("The answer is 42."),
            ]
        )

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

        provider = FakeProvider(
            [
                _tool_response([ToolCall(id="tc_1", name="step_tool", arguments={"arg": "1"})]),
                _tool_response([ToolCall(id="tc_2", name="step_tool", arguments={"arg": "2"})]),
                _text_response("Both steps complete."),
            ]
        )

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

        provider = FakeProvider(
            [
                _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
                _text_response("Done."),
            ]
        )

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

        provider = FakeProvider(
            [
                _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
                _text_response("Permission denied, sorry."),
            ]
        )

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

        provider = FakeProvider(
            [
                _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
                _text_response("Approved and done."),
            ]
        )

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

        provider = FakeProvider(
            [
                _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
                _text_response("User declined."),
            ]
        )

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

        provider = FakeProvider(
            [
                _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
                _text_response("No callback available."),
            ]
        )

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

        provider = FakeProvider(
            [
                _tool_response([ToolCall(id="tc_1", name="failing_tool", arguments={"arg": "x"})]),
                _text_response("Tool failed, I'll try something else."),
            ]
        )

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

        provider = FakeProvider(
            [
                _tool_response([ToolCall(id="tc_1", name="nonexistent", arguments={"arg": "x"})]),
                _text_response("Tool not found, sorry."),
            ]
        )

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

        provider = FakeProvider(
            [
                _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
                _text_response("Done."),
            ]
        )

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


# ---------------------------------------------------------------------------
# Inject queue (message interjection into running loop)
# ---------------------------------------------------------------------------


class TestToolLoopInjection:
    @pytest.mark.asyncio
    async def test_injected_message_appears_in_context(self, tmp_path: Path) -> None:
        """A message queued in inject_queue appears in the next LLM call."""
        handler = AsyncMock(return_value={"ok": True})
        registry = _make_registry(("my_tool", handler))

        inject_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        # Provider: first call returns tool use, second call returns text.
        # We inject a message before the second call.
        call_count = 0

        class InjectingProvider:
            @property
            def provider_name(self) -> str:
                return "fake"

            async def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                model: str | None = None,
            ) -> LLMResponse:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return _tool_response(
                        [ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]
                    )
                # On second call, check that the injected message is present
                user_msgs = [m for m in messages if m.get("role") == "user"]
                assert any("interjected" in m.get("content", "") for m in user_msgs), (
                    "Injected message not found in messages"
                )
                return _text_response("Done with injection.")

        # Queue the message before the loop starts — it'll be drained on iteration 2
        inject_queue.put_nowait({"role": "user", "content": "interjected message"})

        ctx = _make_ctx(tmp_path)
        result = await run_tool_loop(
            provider=InjectingProvider(),
            messages=[{"role": "user", "content": "initial"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
            inject_queue=inject_queue,
        )
        assert result.content == "Done with injection."

    @pytest.mark.asyncio
    async def test_no_injection_when_queue_is_none(self, tmp_path: Path) -> None:
        """inject_queue=None works fine (backward compat)."""
        provider = FakeProvider([_text_response("Hi")])
        ctx = _make_ctx(tmp_path)

        result = await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Hi"}],
            tools=_make_registry(),
            ctx=ctx,
            system_prompt="",
            model="test",
            inject_queue=None,
        )
        assert result.content == "Hi"

    @pytest.mark.asyncio
    async def test_no_injection_when_queue_empty(self, tmp_path: Path) -> None:
        """Empty queue doesn't alter messages."""
        inject_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        provider = FakeProvider([_text_response("Normal")])
        ctx = _make_ctx(tmp_path)

        result = await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Hi"}],
            tools=_make_registry(),
            ctx=ctx,
            system_prompt="",
            model="test",
            inject_queue=inject_queue,
        )
        assert result.content == "Normal"
        # Provider should have received exactly the system + user message
        msgs = provider.call_log[0]["messages"]
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assert len(user_msgs) == 1

    @pytest.mark.asyncio
    async def test_multiple_injected_messages_in_order(self, tmp_path: Path) -> None:
        """3 queued messages appear in FIFO order."""
        inject_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        inject_queue.put_nowait({"role": "user", "content": "first"})
        inject_queue.put_nowait({"role": "user", "content": "second"})
        inject_queue.put_nowait({"role": "user", "content": "third"})

        handler = AsyncMock(return_value={"ok": True})
        registry = _make_registry(("my_tool", handler))

        class CheckingProvider:
            @property
            def provider_name(self) -> str:
                return "fake"

            async def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                model: str | None = None,
            ) -> LLMResponse:
                # All three injected messages should appear in order
                user_contents = [m["content"] for m in messages if m.get("role") == "user"]
                # Should have: "initial", "first", "second", "third"
                assert "first" in user_contents
                assert "second" in user_contents
                assert "third" in user_contents
                idx_first = user_contents.index("first")
                idx_second = user_contents.index("second")
                idx_third = user_contents.index("third")
                assert idx_first < idx_second < idx_third
                return _text_response("All injected.")

        ctx = _make_ctx(tmp_path)
        result = await run_tool_loop(
            provider=CheckingProvider(),
            messages=[{"role": "user", "content": "initial"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
            inject_queue=inject_queue,
        )
        assert result.content == "All injected."


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

        provider = FakeProvider(
            [
                _tool_response(
                    [
                        ToolCall(
                            id="tc_1",
                            name="create_file",
                            arguments={"path": "test.txt", "content": "hello"},
                        )
                    ]
                ),
                _text_response("Done."),
            ]
        )

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


# ---------------------------------------------------------------------------
# on_event callback
# ---------------------------------------------------------------------------


class TestToolLoopOnEvent:
    @pytest.mark.asyncio
    async def test_on_event_fires_for_simple_tool_scenario(self, tmp_path: Path) -> None:
        """Recording callback captures expected event sequence."""
        handler = AsyncMock(return_value={"ok": True})
        registry = _make_registry(("my_tool", handler))

        provider = FakeProvider(
            [
                _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
                _text_response("Done."),
            ]
        )

        events: list[ToolLoopEvent] = []

        async def recorder(event: ToolLoopEvent) -> None:
            events.append(event)

        ctx = _make_ctx(tmp_path)
        await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Do it"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
            on_event=recorder,
        )

        types = [e.type for e in events]
        assert types == [
            ToolLoopEventType.LLM_CALL_START,
            ToolLoopEventType.LLM_CALL_COMPLETE,
            ToolLoopEventType.TOOL_CALL_START,
            ToolLoopEventType.TOOL_CALL_COMPLETE,
            ToolLoopEventType.LLM_CALL_START,
            ToolLoopEventType.LLM_CALL_COMPLETE,
            ToolLoopEventType.LOOP_COMPLETE,
        ]

    @pytest.mark.asyncio
    async def test_on_event_error_logged_not_raised(self, tmp_path: Path) -> None:
        """If the callback raises, the loop still completes."""

        async def bad_callback(event: ToolLoopEvent) -> None:
            raise RuntimeError("callback boom")

        provider = FakeProvider([_text_response("Hi")])
        ctx = _make_ctx(tmp_path)

        result = await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Hi"}],
            tools=_make_registry(),
            ctx=ctx,
            system_prompt="",
            model="test",
            on_event=bad_callback,
        )

        assert result.content == "Hi"

    @pytest.mark.asyncio
    async def test_on_event_none_backward_compat(self, tmp_path: Path) -> None:
        """on_event=None works fine (backward compat)."""
        provider = FakeProvider([_text_response("Hi")])
        ctx = _make_ctx(tmp_path)

        result = await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Hi"}],
            tools=_make_registry(),
            ctx=ctx,
            system_prompt="",
            model="test",
            on_event=None,
        )

        assert result.content == "Hi"

    @pytest.mark.asyncio
    async def test_on_event_usage_delta_correct(self, tmp_path: Path) -> None:
        """LLM_CALL_COMPLETE carries per-call usage delta."""
        handler = AsyncMock(return_value={"ok": True})
        registry = _make_registry(("my_tool", handler))

        provider = FakeProvider(
            [
                _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
                _text_response("Done."),
            ]
        )

        events: list[ToolLoopEvent] = []

        async def recorder(event: ToolLoopEvent) -> None:
            events.append(event)

        ctx = _make_ctx(tmp_path)
        await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Do it"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
            on_event=recorder,
        )

        complete_events = [e for e in events if e.type == ToolLoopEventType.LLM_CALL_COMPLETE]
        assert len(complete_events) == 2
        # First call: _tool_response uses 20 in / 15 out
        assert complete_events[0].usage_delta is not None
        assert complete_events[0].usage_delta.input_tokens == 20
        assert complete_events[0].usage_delta.output_tokens == 15
        # Second call: _text_response uses 10 in / 5 out
        assert complete_events[1].usage_delta is not None
        assert complete_events[1].usage_delta.input_tokens == 10
        assert complete_events[1].usage_delta.output_tokens == 5
        # Total usage on second complete event
        assert complete_events[1].total_usage is not None
        assert complete_events[1].total_usage.input_tokens == 30
        assert complete_events[1].total_usage.output_tokens == 20

    @pytest.mark.asyncio
    async def test_on_event_iteration_count(self, tmp_path: Path) -> None:
        """Events carry the correct iteration number."""
        handler = AsyncMock(return_value={"ok": True})
        registry = _make_registry(("my_tool", handler))

        provider = FakeProvider(
            [
                _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
                _text_response("Done."),
            ]
        )

        events: list[ToolLoopEvent] = []

        async def recorder(event: ToolLoopEvent) -> None:
            events.append(event)

        ctx = _make_ctx(tmp_path)
        await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Do it"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
            on_event=recorder,
        )

        # First iteration events
        iter1_events = [e for e in events if e.iteration == 1]
        assert len(iter1_events) == 4  # LLM_START, LLM_COMPLETE, TOOL_START, TOOL_COMPLETE
        # Second iteration events
        iter2_events = [e for e in events if e.iteration == 2]
        assert len(iter2_events) == 3  # LLM_START, LLM_COMPLETE, LOOP_COMPLETE

    @pytest.mark.asyncio
    async def test_on_event_tools_used_accumulates(self, tmp_path: Path) -> None:
        """Two different tools → both appear in tools_used."""
        h1 = AsyncMock(return_value={"ok": True})
        h2 = AsyncMock(return_value={"ok": True})
        registry = _make_registry(("tool_a", h1), ("tool_b", h2))

        provider = FakeProvider(
            [
                _tool_response(
                    [
                        ToolCall(id="tc_1", name="tool_a", arguments={"arg": "x"}),
                        ToolCall(id="tc_2", name="tool_b", arguments={"arg": "y"}),
                    ]
                ),
                _text_response("Done."),
            ]
        )

        events: list[ToolLoopEvent] = []

        async def recorder(event: ToolLoopEvent) -> None:
            events.append(event)

        ctx = _make_ctx(tmp_path)
        await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Do both"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
            on_event=recorder,
        )

        loop_complete = [e for e in events if e.type == ToolLoopEventType.LOOP_COMPLETE]
        assert len(loop_complete) == 1
        assert set(loop_complete[0].tools_used) == {"tool_a", "tool_b"}

    @pytest.mark.asyncio
    async def test_on_event_tool_call_start_has_name(self, tmp_path: Path) -> None:
        """TOOL_CALL_START event carries the tool_name."""
        handler = AsyncMock(return_value={"ok": True})
        registry = _make_registry(("my_tool", handler))

        provider = FakeProvider(
            [
                _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
                _text_response("Done."),
            ]
        )

        events: list[ToolLoopEvent] = []

        async def recorder(event: ToolLoopEvent) -> None:
            events.append(event)

        ctx = _make_ctx(tmp_path)
        await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Do it"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
            on_event=recorder,
        )

        start_events = [e for e in events if e.type == ToolLoopEventType.TOOL_CALL_START]
        assert len(start_events) == 1
        assert start_events[0].tool_name == "my_tool"


# ---------------------------------------------------------------------------
# Web search integration
# ---------------------------------------------------------------------------


class TestWebSearchIntegration:
    @pytest.mark.asyncio
    async def test_web_search_tool_injected_for_anthropic(self, tmp_path: Path) -> None:
        """Web search tool spec is appended to tool_schemas for Anthropic providers."""

        class AnthropicFake:
            @property
            def provider_name(self) -> str:
                return "anthropic"

            async def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                model: str | None = None,
            ) -> LLMResponse:
                # Verify web search tool is in the tools list
                assert tools is not None
                web_tools = [t for t in tools if t.get("type") == "web_search_20250305"]
                assert len(web_tools) == 1
                assert web_tools[0]["name"] == "web_search"
                assert web_tools[0]["max_uses"] == 5
                return _text_response("Done.")

        ctx = _make_ctx(tmp_path)
        ctx.profile = _open_profile()

        result = await run_tool_loop(
            provider=AnthropicFake(),
            messages=[{"role": "user", "content": "Search for news"}],
            tools=_make_registry(),
            ctx=ctx,
            system_prompt="",
            model="claude-sonnet-4-20250514",
            web_search_enabled=True,
        )
        assert result.content == "Done."

    @pytest.mark.asyncio
    async def test_web_search_not_injected_for_openai(self, tmp_path: Path) -> None:
        """Web search tool is NOT injected for OpenAI providers."""

        class OpenAIFake:
            @property
            def provider_name(self) -> str:
                return "openai"

            async def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                model: str | None = None,
            ) -> LLMResponse:
                # Verify no web search tool
                if tools:
                    web_tools = [t for t in tools if t.get("type") == "web_search_20250305"]
                    assert len(web_tools) == 0
                return _text_response("Done.")

        ctx = _make_ctx(tmp_path)
        ctx.profile = _open_profile()

        result = await run_tool_loop(
            provider=OpenAIFake(),
            messages=[{"role": "user", "content": "Search for news"}],
            tools=_make_registry(),
            ctx=ctx,
            system_prompt="",
            model="gpt-4o",
            web_search_enabled=True,
        )
        assert result.content == "Done."

    @pytest.mark.asyncio
    async def test_web_search_not_injected_when_disabled(self, tmp_path: Path) -> None:
        """web_search_enabled=False means no web search tool."""
        provider = FakeProvider([_text_response("Done.")])
        ctx = _make_ctx(tmp_path)
        ctx.profile = _open_profile()

        result = await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Hi"}],
            tools=_make_registry(),
            ctx=ctx,
            system_prompt="",
            model="test",
            web_search_enabled=False,
        )
        # Check that no web_search_20250305 tool was passed
        if provider.call_log[0]["tools"]:
            for t in provider.call_log[0]["tools"]:
                assert t.get("type") != "web_search_20250305"
        assert result.content == "Done."

    @pytest.mark.asyncio
    async def test_web_search_permission_denied_skips(self, tmp_path: Path) -> None:
        """If permission denies web_search, it's not injected."""

        class AnthropicFake:
            @property
            def provider_name(self) -> str:
                return "anthropic"

            async def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                model: str | None = None,
            ) -> LLMResponse:
                # Should NOT have web search tool
                if tools:
                    web_tools = [t for t in tools if t.get("type") == "web_search_20250305"]
                    assert len(web_tools) == 0
                return _text_response("Done.")

        ctx = _make_ctx(tmp_path)
        ctx.profile = _deny_profile()

        result = await run_tool_loop(
            provider=AnthropicFake(),
            messages=[{"role": "user", "content": "Search"}],
            tools=_make_registry(),
            ctx=ctx,
            system_prompt="",
            model="test",
            web_search_enabled=True,
        )
        assert result.content == "Done."

    @pytest.mark.asyncio
    async def test_web_search_permission_ask_approved(self, tmp_path: Path) -> None:
        """If permission is ASK and user approves, web search is injected."""

        class AnthropicFake:
            @property
            def provider_name(self) -> str:
                return "anthropic"

            async def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                model: str | None = None,
            ) -> LLMResponse:
                assert tools is not None
                web_tools = [t for t in tools if t.get("type") == "web_search_20250305"]
                assert len(web_tools) == 1
                return _text_response("Done.")

        ctx = _make_ctx(tmp_path)
        ctx.profile = _ask_profile()

        ask_callback = AsyncMock(return_value=True)

        result = await run_tool_loop(
            provider=AnthropicFake(),
            messages=[{"role": "user", "content": "Search"}],
            tools=_make_registry(),
            ctx=ctx,
            system_prompt="",
            model="test",
            ask_callback=ask_callback,
            web_search_enabled=True,
        )
        assert result.content == "Done."
        ask_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_web_search_permission_ask_denied(self, tmp_path: Path) -> None:
        """If permission is ASK and user denies, web search is NOT injected."""

        class AnthropicFake:
            @property
            def provider_name(self) -> str:
                return "anthropic"

            async def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                model: str | None = None,
            ) -> LLMResponse:
                if tools:
                    web_tools = [t for t in tools if t.get("type") == "web_search_20250305"]
                    assert len(web_tools) == 0
                return _text_response("Done.")

        ctx = _make_ctx(tmp_path)
        ctx.profile = _ask_profile()

        ask_callback = AsyncMock(return_value=False)

        result = await run_tool_loop(
            provider=AnthropicFake(),
            messages=[{"role": "user", "content": "Search"}],
            tools=_make_registry(),
            ctx=ctx,
            system_prompt="",
            model="test",
            ask_callback=ask_callback,
            web_search_enabled=True,
        )
        assert result.content == "Done."

    @pytest.mark.asyncio
    async def test_web_search_anthropic_content_preserved_on_assistant_msg(
        self, tmp_path: Path
    ) -> None:
        """When response has _raw_content, assistant_msg gets _anthropic_content."""
        raw_blocks = [
            {"type": "text", "text": "Searching..."},
            {"type": "server_tool_use", "id": "srv_1", "name": "web_search"},
        ]

        class AnthropicFake:
            _call_count = 0

            @property
            def provider_name(self) -> str:
                return "anthropic"

            async def chat(
                self,
                messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None = None,
                model: str | None = None,
            ) -> LLMResponse:
                self._call_count += 1
                if self._call_count == 1:
                    resp = _tool_response(
                        [ToolCall(id="tc_1", name="create_file", arguments={"arg": "x"})]
                    )
                    resp._raw_content = raw_blocks
                    return resp
                # On second call, verify _anthropic_content was passed through
                asst_msgs = [m for m in messages if m.get("role") == "assistant"]
                assert any(
                    m.get("_anthropic_content") == raw_blocks for m in asst_msgs
                ), "Expected _anthropic_content on assistant message"
                return _text_response("Done.")

        handler = AsyncMock(return_value={"ok": True})
        registry = _make_registry(("create_file", handler))

        ctx = _make_ctx(tmp_path)

        result = await run_tool_loop(
            provider=AnthropicFake(),
            messages=[{"role": "user", "content": "Do it"}],
            tools=registry,
            ctx=ctx,
            system_prompt="",
            model="test",
        )
        assert result.content == "Done."


# ---------------------------------------------------------------------------
# Mid-loop message truncation
# ---------------------------------------------------------------------------


class TestTruncateToolLoopMessages:
    def test_empty_messages(self) -> None:
        assert _truncate_tool_loop_messages([], 1000) == []

    def test_preserves_system_messages(self) -> None:
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = _truncate_tool_loop_messages(msgs, 100_000)
        assert result[0]["role"] == "system"
        assert len(result) == 3

    def test_keeps_recent_messages_when_budget_tight(self) -> None:
        msgs = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Old " * 1000},
            {"role": "assistant", "content": "Old reply " * 1000},
            {"role": "user", "content": "Recent"},
            {"role": "assistant", "content": "Recent reply"},
        ]
        result = _truncate_tool_loop_messages(msgs, 50)
        assert result[0]["role"] == "system"
        assert result[-1]["content"] == "Recent reply"
        # Old messages should have been dropped
        assert not any("Old " * 100 in m.get("content", "") for m in result)

    def test_atomic_tool_call_block(self) -> None:
        """Assistant+tool messages stay together as an atomic block."""
        msgs = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Do it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc_1", "name": "bash", "arguments": {"command": "ls"}}],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "file1.py\nfile2.py"},
            {"role": "assistant", "content": "Done!"},
        ]
        result = _truncate_tool_loop_messages(msgs, 100_000)
        # All messages should be preserved
        assert len(result) == 5

    def test_atomic_block_not_split(self) -> None:
        """When budget is tight, an assistant+tool block is either kept or dropped as a unit."""
        # Create a large tool result that forms an atomic block
        big_result = "x" * 4000  # ~1000 tokens
        msgs = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Start"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc_1", "name": "view", "arguments": {"path": "big.py"}}],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": big_result},
            {"role": "user", "content": "Recent msg"},
            {"role": "assistant", "content": "Final answer"},
        ]
        # Budget: enough for system + recent user + assistant but not the big tool block
        result = _truncate_tool_loop_messages(msgs, 30)
        # The big tool block should be dropped as a unit
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 0
        # But recent messages should be kept
        assert result[-1]["content"] == "Final answer"

    def test_multiple_tool_blocks_oldest_dropped(self) -> None:
        """With multiple tool blocks, oldest are dropped first."""
        msgs = [
            {"role": "system", "content": "S"},
            # Block 1 (old)
            {"role": "user", "content": "Task 1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc_1", "name": "t", "arguments": {"a": "b"}}],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "Result 1 " * 100},
            # Block 2 (recent)
            {"role": "user", "content": "Task 2"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc_2", "name": "t", "arguments": {"a": "c"}}],
            },
            {"role": "tool", "tool_call_id": "tc_2", "content": "Result 2"},
            {"role": "assistant", "content": "Done"},
        ]
        # Budget enough for system + block 2 + final but not block 1
        result = _truncate_tool_loop_messages(msgs, 80)
        # Block 2 should be kept
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert any("Result 2" in m["content"] for m in tool_msgs)
        # Block 1 (the old big one) should be dropped
        assert not any("Result 1" in m.get("content", "") for m in result)

    def test_system_only_when_budget_tiny(self) -> None:
        msgs = [
            {"role": "system", "content": "Prompt."},
            {"role": "user", "content": "A very long message " * 500},
        ]
        result = _truncate_tool_loop_messages(msgs, 5)
        assert result[0]["role"] == "system"

    def test_tool_calls_with_anthropic_content(self) -> None:
        """Messages with _anthropic_content are estimated properly."""
        msgs = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "Let me search...",
                "tool_calls": [{"id": "tc_1", "name": "bash", "arguments": {"command": "ls"}}],
                "_anthropic_content": [
                    {"type": "text", "text": "Let me search..."},
                    {"type": "tool_use", "id": "tc_1", "name": "bash"},
                ],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "result"},
            {"role": "assistant", "content": "Done"},
        ]
        # Should not crash and should handle the messages
        result = _truncate_tool_loop_messages(msgs, 100_000)
        assert len(result) == 5


class TestToolLoopMidLoopTruncation:
    @pytest.mark.asyncio
    async def test_large_tool_results_dont_blow_up_context(self, tmp_path: Path) -> None:
        """Tool results that push past MAX_INPUT_TOKENS get truncated on next iteration."""
        # Return a massive result from the first tool call
        big_result = "x" * 800_000  # ~200K tokens, will exceed budget
        handler = AsyncMock(return_value=big_result)
        registry = _make_registry(("my_tool", handler))

        provider = FakeProvider(
            [
                _tool_response([ToolCall(id="tc_1", name="my_tool", arguments={"arg": "x"})]),
                _text_response("Done despite big result."),
            ]
        )

        ctx = _make_ctx(tmp_path)
        result = await run_tool_loop(
            provider=provider,
            messages=[{"role": "user", "content": "Do it"}],
            tools=registry,
            ctx=ctx,
            system_prompt="System",
            model="claude-sonnet-4-20250514",
        )

        # Loop should complete successfully (truncation prevented oversize context)
        assert result.content == "Done despite big result."
        assert result.iterations == 2
