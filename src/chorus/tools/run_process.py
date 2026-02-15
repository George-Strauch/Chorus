"""Process runner tools — run_concurrent and run_background."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from chorus.process.callback_builder import build_callbacks_from_instructions
from chorus.process.models import ProcessType
from chorus.tools.bash import CommandBlockedError, _check_blocklist

if TYPE_CHECKING:
    from pathlib import Path

    from chorus.permissions.engine import PermissionProfile
    from chorus.process.manager import ProcessManager

logger = logging.getLogger("chorus.tools.run_process")


async def run_concurrent(
    command: str,
    instructions: str = "",
    *,
    workspace: Path,
    profile: PermissionProfile,
    agent_name: str,
    host_execution: bool = False,
    process_manager: ProcessManager,
    branch_id: int | None = None,
) -> dict[str, Any]:
    """Start a process that runs alongside the active tool loop.

    The branch continues executing. Hooks can inject_context or
    stop_branch.

    Parameters
    ----------
    command:
        Shell command to execute.
    instructions:
        Natural language instructions for what should happen when the
        process produces output or exits.
    """
    # Safety check
    try:
        _check_blocklist(command)
    except CommandBlockedError as exc:
        return {"error": str(exc)}

    # Build callbacks from NL instructions
    callbacks = await build_callbacks_from_instructions(
        instructions=instructions,
        command=command,
    )

    # Spawn process
    tracked = await process_manager.spawn(
        command=command,
        workspace=workspace,
        agent_name=agent_name,
        process_type=ProcessType.CONCURRENT,
        callbacks=callbacks,
        context=instructions,
        spawned_by_branch=branch_id,
    )

    return {
        "pid": tracked.pid,
        "status": "running",
        "type": "concurrent",
        "callbacks": [cb.to_dict() for cb in tracked.callbacks],
        "message": (
            f"Process started (PID {tracked.pid}). "
            f"It runs alongside this branch. "
            f"{len(callbacks)} callback(s) configured."
        ),
    }


async def run_background(
    command: str,
    instructions: str = "",
    model: str | None = None,
    *,
    workspace: Path,
    profile: PermissionProfile,
    agent_name: str,
    host_execution: bool = False,
    process_manager: ProcessManager,
) -> dict[str, Any]:
    """Start a process that outlives the current branch.

    Hooks always spawn new branches. Shows a live Discord embed.

    Parameters
    ----------
    command:
        Shell command to execute.
    instructions:
        Natural language instructions for what should happen when the
        process produces output or exits.
    model:
        Optional model override for hook-spawned branches.
    """
    # Safety check
    try:
        _check_blocklist(command)
    except CommandBlockedError as exc:
        return {"error": str(exc)}

    # Build callbacks from NL instructions
    callbacks = await build_callbacks_from_instructions(
        instructions=instructions,
        command=command,
    )

    # Spawn process
    tracked = await process_manager.spawn(
        command=command,
        workspace=workspace,
        agent_name=agent_name,
        process_type=ProcessType.BACKGROUND,
        callbacks=callbacks,
        context=instructions,
        model_for_hooks=model,
    )

    return {
        "pid": tracked.pid,
        "status": "running",
        "type": "background",
        "callbacks": [cb.to_dict() for cb in tracked.callbacks],
        "message": (
            f"Background process started (PID {tracked.pid}). "
            f"It will continue after this branch ends. "
            f"{len(callbacks)} callback(s) configured."
        ),
    }
