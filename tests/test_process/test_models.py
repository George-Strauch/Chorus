"""Tests for process management data models."""

from __future__ import annotations

import json
from collections import deque

from chorus.process.models import (
    CallbackAction,
    ExitFilter,
    HookTrigger,
    ProcessCallback,
    ProcessStatus,
    ProcessType,
    TrackedProcess,
    TriggerType,
)

# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_process_status_values(self) -> None:
        assert ProcessStatus.RUNNING.value == "running"
        assert ProcessStatus.EXITED.value == "exited"
        assert ProcessStatus.KILLED.value == "killed"
        assert ProcessStatus.LOST.value == "lost"

    def test_process_type_values(self) -> None:
        assert ProcessType.CONCURRENT.value == "concurrent"
        assert ProcessType.BACKGROUND.value == "background"

    def test_trigger_type_values(self) -> None:
        assert TriggerType.ON_EXIT.value == "on_exit"
        assert TriggerType.ON_OUTPUT_MATCH.value == "on_output_match"
        assert TriggerType.ON_TIMEOUT.value == "on_timeout"

    def test_exit_filter_values(self) -> None:
        assert ExitFilter.ANY.value == "any"
        assert ExitFilter.SUCCESS.value == "success"
        assert ExitFilter.FAILURE.value == "failure"

    def test_callback_action_values(self) -> None:
        assert CallbackAction.STOP_PROCESS.value == "stop_process"
        assert CallbackAction.STOP_BRANCH.value == "stop_branch"
        assert CallbackAction.INJECT_CONTEXT.value == "inject_context"
        assert CallbackAction.SPAWN_BRANCH.value == "spawn_branch"


# ---------------------------------------------------------------------------
# HookTrigger tests
# ---------------------------------------------------------------------------


class TestHookTrigger:
    def test_on_exit_trigger(self) -> None:
        t = HookTrigger(type=TriggerType.ON_EXIT, exit_filter=ExitFilter.FAILURE)
        assert t.type == TriggerType.ON_EXIT
        assert t.exit_filter == ExitFilter.FAILURE
        assert t.pattern is None
        assert t.compiled_pattern is None

    def test_on_output_match_trigger(self) -> None:
        t = HookTrigger(type=TriggerType.ON_OUTPUT_MATCH, pattern=r"error|ERROR")
        assert t.pattern == r"error|ERROR"
        cp = t.compiled_pattern
        assert cp is not None
        assert cp.search("ERROR: something failed")
        assert not cp.search("all good")

    def test_compiled_pattern_cached(self) -> None:
        t = HookTrigger(type=TriggerType.ON_OUTPUT_MATCH, pattern=r"\d+")
        p1 = t.compiled_pattern
        p2 = t.compiled_pattern
        assert p1 is p2

    def test_on_timeout_trigger(self) -> None:
        t = HookTrigger(type=TriggerType.ON_TIMEOUT, timeout_seconds=30.0)
        assert t.timeout_seconds == 30.0

    def test_to_dict_minimal(self) -> None:
        t = HookTrigger(type=TriggerType.ON_EXIT)
        d = t.to_dict()
        assert d == {"type": "on_exit"}

    def test_to_dict_full(self) -> None:
        t = HookTrigger(
            type=TriggerType.ON_OUTPUT_MATCH,
            exit_filter=ExitFilter.SUCCESS,
            pattern="done",
            timeout_seconds=10.0,
        )
        d = t.to_dict()
        assert d == {
            "type": "on_output_match",
            "exit_filter": "success",
            "pattern": "done",
            "timeout_seconds": 10.0,
        }

    def test_from_dict(self) -> None:
        data = {"type": "on_exit", "exit_filter": "failure"}
        t = HookTrigger.from_dict(data)
        assert t.type == TriggerType.ON_EXIT
        assert t.exit_filter == ExitFilter.FAILURE

    def test_from_dict_defaults(self) -> None:
        data = {"type": "on_timeout"}
        t = HookTrigger.from_dict(data)
        assert t.exit_filter == ExitFilter.ANY
        assert t.pattern is None

    def test_roundtrip(self) -> None:
        original = HookTrigger(
            type=TriggerType.ON_OUTPUT_MATCH,
            pattern=r"test\d+",
            timeout_seconds=5.0,
        )
        d = original.to_dict()
        restored = HookTrigger.from_dict(d)
        assert restored.type == original.type
        assert restored.pattern == original.pattern
        assert restored.timeout_seconds == original.timeout_seconds


# ---------------------------------------------------------------------------
# ProcessCallback tests
# ---------------------------------------------------------------------------


class TestProcessCallback:
    def test_basic_construction(self) -> None:
        trigger = HookTrigger(type=TriggerType.ON_EXIT)
        cb = ProcessCallback(
            trigger=trigger,
            action=CallbackAction.SPAWN_BRANCH,
            context_message="Process exited",
        )
        assert cb.action == CallbackAction.SPAWN_BRANCH
        assert cb.context_message == "Process exited"
        assert cb.fire_count == 0
        assert cb.max_fires == 1

    def test_exhausted(self) -> None:
        trigger = HookTrigger(type=TriggerType.ON_EXIT)
        cb = ProcessCallback(trigger=trigger, action=CallbackAction.STOP_PROCESS, max_fires=2)
        assert not cb.exhausted
        cb.fire_count = 1
        assert not cb.exhausted
        cb.fire_count = 2
        assert cb.exhausted

    def test_to_dict(self) -> None:
        trigger = HookTrigger(type=TriggerType.ON_EXIT, exit_filter=ExitFilter.FAILURE)
        cb = ProcessCallback(
            trigger=trigger,
            action=CallbackAction.INJECT_CONTEXT,
            context_message="Failed!",
            output_delay_seconds=1.5,
            max_fires=3,
            fire_count=1,
        )
        d = cb.to_dict()
        assert d["action"] == "inject_context"
        assert d["context_message"] == "Failed!"
        assert d["output_delay_seconds"] == 1.5
        assert d["max_fires"] == 3
        assert d["fire_count"] == 1
        assert d["trigger"]["type"] == "on_exit"

    def test_from_dict(self) -> None:
        data = {
            "trigger": {"type": "on_output_match", "pattern": "error"},
            "action": "stop_process",
            "context_message": "",
            "output_delay_seconds": 0.0,
            "max_fires": 1,
            "fire_count": 0,
        }
        cb = ProcessCallback.from_dict(data)
        assert cb.action == CallbackAction.STOP_PROCESS
        assert cb.trigger.type == TriggerType.ON_OUTPUT_MATCH
        assert cb.trigger.pattern == "error"

    def test_roundtrip(self) -> None:
        original = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_EXIT, exit_filter=ExitFilter.SUCCESS),
            action=CallbackAction.SPAWN_BRANCH,
            context_message="All done",
            output_delay_seconds=2.0,
            max_fires=5,
            fire_count=2,
        )
        restored = ProcessCallback.from_dict(original.to_dict())
        assert restored.action == original.action
        assert restored.trigger.type == original.trigger.type
        assert restored.trigger.exit_filter == original.trigger.exit_filter
        assert restored.context_message == original.context_message
        assert restored.max_fires == original.max_fires
        assert restored.fire_count == original.fire_count


# ---------------------------------------------------------------------------
# TrackedProcess tests
# ---------------------------------------------------------------------------


class TestTrackedProcess:
    def _make_tracked(self, **overrides: object) -> TrackedProcess:
        defaults: dict[str, object] = {
            "pid": 12345,
            "command": "python server.py",
            "working_directory": "/tmp/workspace",
            "agent_name": "test-agent",
            "started_at": "2026-02-14T12:00:00+00:00",
            "process_type": ProcessType.BACKGROUND,
        }
        defaults.update(overrides)
        return TrackedProcess(**defaults)  # type: ignore[arg-type]

    def test_basic_construction(self) -> None:
        p = self._make_tracked()
        assert p.pid == 12345
        assert p.status == ProcessStatus.RUNNING
        assert p.exit_code is None
        assert len(p.callbacks) == 0
        assert isinstance(p.rolling_tail, deque)

    def test_to_dict(self) -> None:
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_EXIT),
            action=CallbackAction.SPAWN_BRANCH,
        )
        p = self._make_tracked(callbacks=[cb], context="test context")
        d = p.to_dict()
        assert d["pid"] == 12345
        assert d["process_type"] == "background"
        assert d["status"] == "running"
        assert len(d["callbacks"]) == 1
        assert d["context"] == "test context"
        assert d["rolling_tail"] == []

    def test_from_dict(self) -> None:
        data = {
            "pid": 999,
            "command": "npm start",
            "working_directory": "/workspace",
            "agent_name": "web-agent",
            "started_at": "2026-01-01T00:00:00",
            "process_type": "concurrent",
            "status": "exited",
            "exit_code": 0,
            "callbacks": [],
            "rolling_tail": ["line1", "line2"],
        }
        p = TrackedProcess.from_dict(data)
        assert p.pid == 999
        assert p.process_type == ProcessType.CONCURRENT
        assert p.status == ProcessStatus.EXITED
        assert p.exit_code == 0
        assert list(p.rolling_tail) == ["line1", "line2"]

    def test_from_db_row(self) -> None:
        row = {
            "pid": 42,
            "command": "sleep 100",
            "working_directory": "/w",
            "agent_name": "a",
            "started_at": "2026-01-01T00:00:00",
            "process_type": "background",
            "spawned_by_branch": 3,
            "stdout_log": "/logs/stdout.log",
            "stderr_log": "/logs/stderr.log",
            "status": "running",
            "exit_code": None,
            "callbacks_json": json.dumps([{
                "trigger": {"type": "on_exit"},
                "action": "spawn_branch",
                "context_message": "done",
                "output_delay_seconds": 0.0,
                "max_fires": 1,
                "fire_count": 0,
            }]),
            "context_json": "some context",
            "model_for_hooks": "claude-haiku-4-5-20251001",
            "hook_recursion_depth": 1,
            "discord_message_id": 12345678,
        }
        p = TrackedProcess.from_db_row(row)
        assert p.pid == 42
        assert p.spawned_by_branch == 3
        assert len(p.callbacks) == 1
        assert p.callbacks[0].action == CallbackAction.SPAWN_BRANCH
        assert p.model_for_hooks == "claude-haiku-4-5-20251001"
        assert p.hook_recursion_depth == 1

    def test_roundtrip(self) -> None:
        cb = ProcessCallback(
            trigger=HookTrigger(type=TriggerType.ON_OUTPUT_MATCH, pattern="fail"),
            action=CallbackAction.STOP_PROCESS,
            max_fires=3,
        )
        original = self._make_tracked(
            callbacks=[cb],
            model_for_hooks="gpt-4o-mini",
            hook_recursion_depth=2,
        )
        original.rolling_tail.extend(["a", "b", "c"])
        d = original.to_dict()
        restored = TrackedProcess.from_dict(d)
        assert restored.pid == original.pid
        assert restored.model_for_hooks == original.model_for_hooks
        assert restored.hook_recursion_depth == original.hook_recursion_depth
        assert len(restored.callbacks) == 1
        assert list(restored.rolling_tail) == ["a", "b", "c"]

    def test_rolling_tail_maxlen(self) -> None:
        p = self._make_tracked()
        for i in range(150):
            p.rolling_tail.append(f"line {i}")
        assert len(p.rolling_tail) == 100
        assert p.rolling_tail[0] == "line 50"
