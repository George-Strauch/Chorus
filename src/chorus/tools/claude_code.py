"""Claude Code SDK integration — delegate coding tasks to Claude Code.

Uses the ``claude-agent-sdk`` package to invoke Claude Code as a tool.
Each invocation is a one-shot ``query()`` call — the bot passes a task
description, Claude Code does the work, results come back.

Gracefully degrades when the SDK is not installed: ``is_claude_code_available()``
returns False and the tool is not registered.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from chorus.permissions.engine import PermissionProfile

logger = logging.getLogger("chorus.tools.claude_code")

# ---------------------------------------------------------------------------
# SDK availability check
# ---------------------------------------------------------------------------

_sdk_available: bool = False

try:
    import claude_agent_sdk  # noqa: F401

    _sdk_available = True
except (ImportError, ModuleNotFoundError):
    _sdk_available = False


def is_claude_code_available() -> bool:
    """Check if the claude-agent-sdk package is importable."""
    return _sdk_available


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------

_MAX_OUTPUT_LEN = 50_000


def _truncate_output(text: str, max_len: int = _MAX_OUTPUT_LEN) -> str:
    """Truncate output to *max_len* characters if needed."""
    if len(text) <= max_len:
        return text
    header = f"[truncated: showing last {max_len} chars of {len(text)} chars]\n"
    return header + text[-max_len:]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ClaudeCodeResult:
    """Structured result from a Claude Code execution."""

    task: str
    success: bool
    output: str
    cost_usd: float | None
    duration_ms: int
    num_turns: int
    error: str | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "success": self.success,
            "output": self.output,
            "cost_usd": self.cost_usd,
            "duration_ms": self.duration_ms,
            "num_turns": self.num_turns,
            "error": self.error,
            "session_id": self.session_id,
        }


# ---------------------------------------------------------------------------
# SDK query wrapper (separated for easy mocking)
# ---------------------------------------------------------------------------


async def _run_sdk_query(
    *,
    task: str,
    cwd: Path,
    model: str | None = None,
    max_turns: int | None = None,
    max_budget_usd: float | None = None,
    host_execution: bool = False,
    on_progress: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Run the Claude Agent SDK query() and collect results.

    Returns a dict with output, cost_usd, duration_ms, num_turns, session_id, is_error.

    Parameters
    ----------
    on_progress:
        Optional callback fired on each SDK message with progress info.
        Receives a dict with keys like ``tool_name``, ``tool_input``,
        ``turn``, ``text_preview``.
    """
    from claude_agent_sdk import (
        AgentDefinition,
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        query,
    )

    # Configure a cheap Haiku subagent for read-only research tasks
    agents = {
        "researcher": AgentDefinition(
            description=(
                "Fast, cheap agent for reading files, searching codebases, "
                "listing directories, running simple bash commands, and gathering info."
            ),
            prompt="You are a fast researcher. Read files, search code, run commands.",
            tools=["Read", "Glob", "Grep", "Bash"],
            model="haiku",
        ),
    }

    options_kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "permission_mode": "acceptEdits",
        "allowed_tools": ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task"],
        "agents": agents,
    }
    if model is not None:
        options_kwargs["model"] = model
    if max_turns is not None:
        options_kwargs["max_turns"] = max_turns
    if max_budget_usd is not None:
        options_kwargs["max_budget_usd"] = max_budget_usd

    if not host_execution:
        # Restrict environment — only pass ANTHROPIC_API_KEY
        import os

        env: dict[str, str] = {}
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        options_kwargs["env"] = env

    options = ClaudeAgentOptions(**options_kwargs)

    output_parts: list[str] = []
    result_info: dict[str, Any] = {
        "output": "",
        "cost_usd": None,
        "duration_ms": 0,
        "num_turns": 0,
        "session_id": None,
        "is_error": False,
    }
    turn_count = 0

    async for message in query(prompt=task, options=options):
        # Collect text from assistant messages
        if isinstance(message, AssistantMessage):
            turn_count += 1
            for block in message.content:
                if hasattr(block, "text"):
                    output_parts.append(block.text)
                # Fire progress for tool use blocks
                if on_progress is not None and hasattr(block, "name"):
                    # ToolUseBlock has name and input
                    tool_input = getattr(block, "input", {}) or {}
                    progress_info: dict[str, Any] = {
                        "turn": turn_count,
                        "tool_name": block.name,
                        "tool_input": tool_input,
                    }
                    try:
                        result = on_progress(progress_info)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        logger.debug("on_progress callback error", exc_info=True)
        elif isinstance(message, ResultMessage):
            result_info["duration_ms"] = message.duration_ms
            result_info["num_turns"] = message.num_turns
            result_info["session_id"] = message.session_id
            result_info["cost_usd"] = message.total_cost_usd
            result_info["is_error"] = message.is_error
            if message.result is not None:
                output_parts.append(message.result)

    result_info["output"] = "\n".join(output_parts)
    return result_info


# ---------------------------------------------------------------------------
# Main tool handler
# ---------------------------------------------------------------------------


async def claude_code_execute(
    task: str,
    workspace: Path,
    profile: PermissionProfile,
    agent_name: str = "",
    max_turns: int = 50,
    max_budget_usd: float = 1.0,
    timeout: float = 600,
    model: str | None = None,
    host_execution: bool = False,
    on_tool_progress: Callable[[dict[str, Any]], Any] | None = None,
) -> ClaudeCodeResult:
    """Execute a coding task via Claude Code SDK.

    Parameters
    ----------
    task:
        Natural language description of what to do.
    workspace:
        Agent workspace directory (passed as cwd to Claude Code).
    profile:
        Permission profile (injected by tool loop).
    agent_name:
        Agent identifier (injected).
    max_turns:
        Maximum agentic turns for Claude Code.
    max_budget_usd:
        Maximum cost budget in USD.
    timeout:
        Timeout in seconds for the entire operation.
    model:
        Model to use. Only forwarded if it's an Anthropic model.
    host_execution:
        If True, inherit full host environment. If False, restrict env.
    """
    start = time.monotonic()

    # Check SDK availability
    if not _sdk_available:
        return ClaudeCodeResult(
            task=task,
            success=False,
            output="",
            cost_usd=None,
            duration_ms=0,
            num_turns=0,
            error="Claude Agent SDK (claude-agent-sdk) is not installed. "
            "Install it with: pip install claude-agent-sdk",
        )

    # Only forward Anthropic models to Claude Code (SDK is Anthropic-only)
    forwarded_model: str | None = None
    if model and model.startswith("claude"):
        forwarded_model = model

    try:
        result_info = await asyncio.wait_for(
            _run_sdk_query(
                task=task,
                cwd=workspace,
                model=forwarded_model,
                max_turns=max_turns,
                max_budget_usd=max_budget_usd,
                host_execution=host_execution,
                on_progress=on_tool_progress,
            ),
            timeout=timeout,
        )
    except TimeoutError:
        elapsed = int((time.monotonic() - start) * 1000)
        return ClaudeCodeResult(
            task=task,
            success=False,
            output="",
            cost_usd=None,
            duration_ms=elapsed,
            num_turns=0,
            error=f"Timeout after {timeout}s",
        )
    except Exception as exc:
        elapsed = int((time.monotonic() - start) * 1000)
        return ClaudeCodeResult(
            task=task,
            success=False,
            output="",
            cost_usd=None,
            duration_ms=elapsed,
            num_turns=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    elapsed = int((time.monotonic() - start) * 1000)
    output = _truncate_output(result_info["output"])
    is_error = result_info.get("is_error", False)

    return ClaudeCodeResult(
        task=task,
        success=not is_error,
        output=output,
        cost_usd=result_info.get("cost_usd"),
        duration_ms=result_info.get("duration_ms", elapsed),
        num_turns=result_info.get("num_turns", 0),
        session_id=result_info.get("session_id"),
        error=result_info.get("output", "Claude Code returned an error") if is_error else None,
    )
