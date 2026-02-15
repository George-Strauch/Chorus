"""Process runner tools — run_concurrent, run_background, add_process_hooks."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from chorus.process.callback_builder import build_callbacks_from_instructions
from chorus.process.models import ProcessStatus, ProcessType
from chorus.tools.bash import CommandBlockedError, _check_blocklist

if TYPE_CHECKING:
    from pathlib import Path

    from chorus.permissions.engine import PermissionProfile
    from chorus.process.hooks import HookDispatcher
    from chorus.process.manager import ProcessManager

logger = logging.getLogger("chorus.tools.run_process")


def _resolve_working_directory(
    working_directory: str,
    workspace: Path,
    scope_path: Path | None,
) -> Path:
    """Resolve and validate a working_directory relative to workspace.

    Falls back to workspace if empty. Validates against path traversal
    by checking the resolved path is under workspace or scope_path.
    """
    from pathlib import Path as _Path

    if not working_directory:
        return workspace

    candidate = _Path(working_directory)
    if not candidate.is_absolute():
        candidate = workspace / candidate

    resolved = candidate.resolve()

    # Allow paths under workspace
    ws_resolved = workspace.resolve()
    if str(resolved).startswith(str(ws_resolved)):
        return resolved

    # Allow paths under scope_path if configured
    if scope_path is not None:
        sp_resolved = scope_path.resolve()
        if str(resolved).startswith(str(sp_resolved)):
            return resolved

    raise ValueError(
        f"working_directory {working_directory!r} is outside the allowed paths"
    )


async def run_concurrent(
    command: str,
    instructions: str = "",
    working_directory: str = "",
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
        Shell command to execute. Launch ONE process per independent
        script/command. Do not chain multiple scripts with && — spawn
        separate processes for each.
    instructions:
        Natural language instructions for what should happen when the
        process produces output or exits.
    working_directory:
        Directory to run the command in. Defaults to the agent workspace.
        Use this instead of prefixing commands with ``cd /path &&``.
    """
    # Safety check
    try:
        _check_blocklist(command)
    except CommandBlockedError as exc:
        return {"error": str(exc)}

    # Resolve working directory
    scope_path_str = os.environ.get("SCOPE_PATH")
    scope_path: Path | None = None
    if scope_path_str:
        from pathlib import Path as _Path
        scope_path = _Path(scope_path_str)

    try:
        resolved_ws = _resolve_working_directory(working_directory, workspace, scope_path)
    except ValueError as exc:
        return {"error": str(exc)}

    # Build callbacks from NL instructions
    callbacks = await build_callbacks_from_instructions(
        instructions=instructions,
        command=command,
    )

    # Spawn process
    tracked = await process_manager.spawn(
        command=command,
        workspace=resolved_ws,
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
    working_directory: str = "",
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
        Shell command to execute. Launch ONE process per independent
        script/command. Do not chain multiple scripts with && — spawn
        separate processes for each.
    instructions:
        Natural language instructions for what should happen when the
        process produces output or exits.
    model:
        Optional model override for hook-spawned branches.
    working_directory:
        Directory to run the command in. Defaults to the agent workspace.
        Use this instead of prefixing commands with ``cd /path &&``.
    """
    # Safety check
    try:
        _check_blocklist(command)
    except CommandBlockedError as exc:
        return {"error": str(exc)}

    # Resolve working directory
    scope_path_str = os.environ.get("SCOPE_PATH")
    scope_path: Path | None = None
    if scope_path_str:
        from pathlib import Path as _Path
        scope_path = _Path(scope_path_str)

    try:
        resolved_ws = _resolve_working_directory(working_directory, workspace, scope_path)
    except ValueError as exc:
        return {"error": str(exc)}

    # Build callbacks from NL instructions
    callbacks = await build_callbacks_from_instructions(
        instructions=instructions,
        command=command,
    )

    # Spawn process
    tracked = await process_manager.spawn(
        command=command,
        workspace=resolved_ws,
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


async def add_process_hooks(
    pid: int,
    instructions: str,
    *,
    agent_name: str,
    process_manager: ProcessManager,
    hook_dispatcher: HookDispatcher,
) -> dict[str, Any]:
    """Add hooks to a running process.

    Parameters
    ----------
    pid:
        PID of the running process to add hooks to.
    instructions:
        Natural language instructions for the new hooks.
    """
    tracked = process_manager.get_process(pid)
    if tracked is None:
        return {"error": f"No process found with PID {pid}"}
    if tracked.agent_name != agent_name:
        return {"error": f"Process {pid} belongs to agent '{tracked.agent_name}', not '{agent_name}'"}
    if tracked.status != ProcessStatus.RUNNING:
        return {"error": f"Process {pid} is not running (status: {tracked.status.value})"}

    callbacks = await build_callbacks_from_instructions(
        instructions=instructions,
        command=tracked.command,
    )

    result = await process_manager.add_callbacks(pid, callbacks)
    if result is None:
        return {"error": f"Failed to add callbacks to process {pid} (it may have exited)"}

    hook_dispatcher.start_new_timeout_watchers(pid, callbacks)

    return {
        "pid": pid,
        "added": len(callbacks),
        "total": len(result.callbacks),
        "callbacks": [cb.to_dict() for cb in callbacks],
        "message": (
            f"Added {len(callbacks)} hook(s) to process {pid}. "
            f"Total hooks: {len(result.callbacks)}."
        ),
    }
