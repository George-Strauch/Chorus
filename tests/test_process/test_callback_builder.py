"""Tests for CallbackBuilder — NL→structured callback translation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from chorus.process.callback_builder import (
    _DEFAULT_CALLBACK,
    _parse_callbacks,
    _parse_single_callback,
    build_callbacks_from_instructions,
)
from chorus.process.models import (
    CallbackAction,
    ExitFilter,
    TriggerType,
)

# ---------------------------------------------------------------------------
# _parse_single_callback tests
# ---------------------------------------------------------------------------


class TestParseSingleCallback:
    def test_minimal(self) -> None:
        item = {
            "trigger": {"type": "on_exit"},
            "action": "spawn_branch",
        }
        cb = _parse_single_callback(item, default_output_delay=2.0)
        assert cb.trigger.type == TriggerType.ON_EXIT
        assert cb.action == CallbackAction.SPAWN_BRANCH
        assert cb.output_delay_seconds == 0.0
        assert cb.max_fires == 1

    def test_full(self) -> None:
        item = {
            "trigger": {
                "type": "on_output_match",
                "pattern": r"error|ERROR",
                "exit_filter": "failure",
            },
            "action": "stop_process",
            "context_message": "Error detected",
            "output_delay_seconds": 1.5,
            "max_fires": 3,
        }
        cb = _parse_single_callback(item, default_output_delay=2.0)
        assert cb.trigger.type == TriggerType.ON_OUTPUT_MATCH
        assert cb.trigger.pattern == r"error|ERROR"
        assert cb.action == CallbackAction.STOP_PROCESS
        assert cb.context_message == "Error detected"
        assert cb.output_delay_seconds == 1.5
        assert cb.max_fires == 3

    def test_output_match_default_delay(self) -> None:
        """Output match callbacks get default delay if not specified."""
        item = {
            "trigger": {"type": "on_output_match", "pattern": "test"},
            "action": "inject_context",
        }
        cb = _parse_single_callback(item, default_output_delay=2.0)
        assert cb.output_delay_seconds == 2.0

    def test_on_timeout(self) -> None:
        item = {
            "trigger": {"type": "on_timeout", "timeout_seconds": 300},
            "action": "stop_process",
        }
        cb = _parse_single_callback(item, default_output_delay=2.0)
        assert cb.trigger.type == TriggerType.ON_TIMEOUT
        assert cb.trigger.timeout_seconds == 300


# ---------------------------------------------------------------------------
# _parse_callbacks tests
# ---------------------------------------------------------------------------


class TestParseCallbacks:
    def test_valid_json_array(self) -> None:
        raw = json.dumps([
            {"trigger": {"type": "on_exit"}, "action": "spawn_branch"},
            {"trigger": {"type": "on_output_match", "pattern": "done"}, "action": "inject_context"},
        ])
        cbs = _parse_callbacks(raw, default_output_delay=2.0)
        assert len(cbs) == 2
        assert cbs[0].action == CallbackAction.SPAWN_BRANCH
        assert cbs[1].action == CallbackAction.INJECT_CONTEXT

    def test_markdown_code_fences(self) -> None:
        raw = '```json\n[{"trigger": {"type": "on_exit"}, "action": "spawn_branch"}]\n```'
        cbs = _parse_callbacks(raw, default_output_delay=2.0)
        assert len(cbs) == 1

    def test_single_object(self) -> None:
        """Single object (not array) is accepted."""
        raw = json.dumps(
            {"trigger": {"type": "on_exit"}, "action": "stop_process"}
        )
        cbs = _parse_callbacks(raw, default_output_delay=2.0)
        assert len(cbs) == 1
        assert cbs[0].action == CallbackAction.STOP_PROCESS

    def test_invalid_json(self) -> None:
        cbs = _parse_callbacks("not json at all", default_output_delay=2.0)
        assert cbs == []

    def test_invalid_item_skipped(self) -> None:
        """Invalid items are skipped, valid ones kept."""
        raw = json.dumps([
            {"trigger": {"type": "on_exit"}, "action": "spawn_branch"},
            {"trigger": {"type": "invalid_type"}, "action": "nope"},
        ])
        cbs = _parse_callbacks(raw, default_output_delay=2.0)
        assert len(cbs) == 1


# ---------------------------------------------------------------------------
# build_callbacks_from_instructions tests
# ---------------------------------------------------------------------------


class TestBuildCallbacksFromInstructions:
    @pytest.mark.asyncio
    async def test_empty_instructions_returns_default(self) -> None:
        """Empty instructions return the default callback."""
        cbs = await build_callbacks_from_instructions("")
        assert len(cbs) == 1
        assert cbs[0] is _DEFAULT_CALLBACK

    @pytest.mark.asyncio
    async def test_whitespace_instructions_returns_default(self) -> None:
        cbs = await build_callbacks_from_instructions("   ")
        assert len(cbs) == 1
        assert cbs[0] is _DEFAULT_CALLBACK

    @pytest.mark.asyncio
    async def test_successful_sub_agent_call(self) -> None:
        """Successful sub-agent call returns parsed callbacks."""
        mock_result = AsyncMock()
        mock_result.return_value.success = True
        mock_result.return_value.output = json.dumps([
            {"trigger": {"type": "on_exit", "exit_filter": "failure"}, "action": "spawn_branch",
             "context_message": "Fix the error"},
        ])

        with patch(
            "chorus.process.callback_builder.run_sub_agent",
            mock_result,
        ):
            cbs = await build_callbacks_from_instructions("if it fails, fix it")

        assert len(cbs) == 1
        assert cbs[0].trigger.exit_filter == ExitFilter.FAILURE
        assert cbs[0].action == CallbackAction.SPAWN_BRANCH

    @pytest.mark.asyncio
    async def test_sub_agent_failure_returns_default(self) -> None:
        """Sub-agent failure falls back to default."""
        mock_result = AsyncMock()
        mock_result.return_value.success = False
        mock_result.return_value.error = "No API key"

        with patch(
            "chorus.process.callback_builder.run_sub_agent",
            mock_result,
        ):
            cbs = await build_callbacks_from_instructions("do something")

        assert len(cbs) == 1
        assert cbs[0].action == CallbackAction.SPAWN_BRANCH

    @pytest.mark.asyncio
    async def test_sub_agent_exception_returns_default(self) -> None:
        """Exception during sub-agent call falls back to default."""
        with patch(
            "chorus.process.callback_builder.run_sub_agent",
            side_effect=RuntimeError("boom"),
        ):
            cbs = await build_callbacks_from_instructions("do something")

        assert len(cbs) == 1
        assert cbs[0].action == CallbackAction.SPAWN_BRANCH
