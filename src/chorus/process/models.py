"""Data models for process management — enums and dataclasses."""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProcessStatus(Enum):
    """Lifecycle status of a tracked process."""

    RUNNING = "running"
    EXITED = "exited"
    KILLED = "killed"
    LOST = "lost"


class ProcessType(Enum):
    """How the process relates to the agent's execution."""

    CONCURRENT = "concurrent"
    BACKGROUND = "background"


class TriggerType(Enum):
    """What event fires a callback."""

    ON_EXIT = "on_exit"
    ON_OUTPUT_MATCH = "on_output_match"
    ON_TIMEOUT = "on_timeout"


class ExitFilter(Enum):
    """Which exit codes trigger an on_exit callback."""

    ANY = "any"
    SUCCESS = "success"
    FAILURE = "failure"


class CallbackAction(Enum):
    """What happens when a callback fires."""

    STOP_PROCESS = "stop_process"
    STOP_BRANCH = "stop_branch"
    INJECT_CONTEXT = "inject_context"
    SPAWN_BRANCH = "spawn_branch"
    NOTIFY_CHANNEL = "notify_channel"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HookTrigger:
    """Describes when a callback should fire."""

    type: TriggerType
    exit_filter: ExitFilter = ExitFilter.ANY
    pattern: str | None = None
    timeout_seconds: float | None = None

    # Compiled regex (lazy, not serialized)
    _compiled: re.Pattern[str] | None = field(
        init=False, repr=False, default=None, compare=False,
    )

    @property
    def compiled_pattern(self) -> re.Pattern[str] | None:
        """Return a compiled regex for ON_OUTPUT_MATCH triggers."""
        if self.type != TriggerType.ON_OUTPUT_MATCH or self.pattern is None:
            return None
        if self._compiled is None:
            self._compiled = re.compile(self.pattern)
        return self._compiled

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type.value}
        if self.exit_filter != ExitFilter.ANY:
            d["exit_filter"] = self.exit_filter.value
        if self.pattern is not None:
            d["pattern"] = self.pattern
        if self.timeout_seconds is not None:
            d["timeout_seconds"] = self.timeout_seconds
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HookTrigger:
        return cls(
            type=TriggerType(data["type"]),
            exit_filter=ExitFilter(data.get("exit_filter", "any")),
            pattern=data.get("pattern"),
            timeout_seconds=data.get("timeout_seconds"),
        )


@dataclass
class ProcessCallback:
    """A single callback attached to a process."""

    trigger: HookTrigger
    action: CallbackAction
    context_message: str = ""
    output_delay_seconds: float = 0.0
    max_fires: int = 1
    fire_count: int = 0

    @property
    def exhausted(self) -> bool:
        return self.fire_count >= self.max_fires

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger.to_dict(),
            "action": self.action.value,
            "context_message": self.context_message,
            "output_delay_seconds": self.output_delay_seconds,
            "max_fires": self.max_fires,
            "fire_count": self.fire_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessCallback:
        return cls(
            trigger=HookTrigger.from_dict(data["trigger"]),
            action=CallbackAction(data["action"]),
            context_message=data.get("context_message", ""),
            output_delay_seconds=data.get("output_delay_seconds", 0.0),
            max_fires=data.get("max_fires", 1),
            fire_count=data.get("fire_count", 0),
        )


@dataclass
class TrackedProcess:
    """A process being tracked by the ProcessManager."""

    pid: int
    command: str
    working_directory: str
    agent_name: str
    started_at: str
    process_type: ProcessType
    spawned_by_branch: int | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None
    status: ProcessStatus = ProcessStatus.RUNNING
    exit_code: int | None = None
    callbacks: list[ProcessCallback] = field(default_factory=list)
    context: str = ""
    rolling_tail: deque[str] = field(default_factory=lambda: deque(maxlen=100))
    model_for_hooks: str | None = None
    hook_recursion_depth: int = 0
    discord_message_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "command": self.command,
            "working_directory": self.working_directory,
            "agent_name": self.agent_name,
            "started_at": self.started_at,
            "process_type": self.process_type.value,
            "spawned_by_branch": self.spawned_by_branch,
            "stdout_log": self.stdout_log,
            "stderr_log": self.stderr_log,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "callbacks": [cb.to_dict() for cb in self.callbacks],
            "context": self.context,
            "rolling_tail": list(self.rolling_tail),
            "model_for_hooks": self.model_for_hooks,
            "hook_recursion_depth": self.hook_recursion_depth,
            "discord_message_id": self.discord_message_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackedProcess:
        callbacks = [
            ProcessCallback.from_dict(cb) for cb in data.get("callbacks", [])
        ]
        tail = data.get("rolling_tail", [])
        return cls(
            pid=data["pid"],
            command=data["command"],
            working_directory=data["working_directory"],
            agent_name=data["agent_name"],
            started_at=data["started_at"],
            process_type=ProcessType(data["process_type"]),
            spawned_by_branch=data.get("spawned_by_branch"),
            stdout_log=data.get("stdout_log"),
            stderr_log=data.get("stderr_log"),
            status=ProcessStatus(data.get("status", "running")),
            exit_code=data.get("exit_code"),
            callbacks=callbacks,
            context=data.get("context", ""),
            rolling_tail=deque(tail, maxlen=100),
            model_for_hooks=data.get("model_for_hooks"),
            hook_recursion_depth=data.get("hook_recursion_depth", 0),
            discord_message_id=data.get("discord_message_id"),
        )

    @classmethod
    def from_db_row(cls, row: dict[str, Any]) -> TrackedProcess:
        """Construct from a DB row dict (callbacks/context stored as JSON strings)."""
        callbacks_raw = row.get("callbacks_json") or "[]"
        callbacks = [ProcessCallback.from_dict(cb) for cb in json.loads(callbacks_raw)]
        context = row.get("context_json") or ""
        return cls(
            pid=row["pid"],
            command=row["command"],
            working_directory=row["working_directory"],
            agent_name=row["agent_name"],
            started_at=row["started_at"],
            process_type=ProcessType(row["process_type"]),
            spawned_by_branch=row.get("spawned_by_branch"),
            stdout_log=row.get("stdout_log"),
            stderr_log=row.get("stderr_log"),
            status=ProcessStatus(row.get("status", "running")),
            exit_code=row.get("exit_code"),
            callbacks=callbacks,
            context=context,
            model_for_hooks=row.get("model_for_hooks"),
            hook_recursion_depth=row.get("hook_recursion_depth", 0),
            discord_message_id=row.get("discord_message_id"),
        )
