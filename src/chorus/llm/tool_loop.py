"""Agentic tool use loop — the core execution engine.

Sends messages + tools to an LLM, executes tool calls with permission
checks, feeds results back, and loops until the LLM produces a final
text response or the iteration cap is reached.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from chorus.llm.providers import (
    LLMProvider,
    ToolCall,
    Usage,
    tools_to_anthropic,
    tools_to_openai,
)
from chorus.permissions.engine import PermissionResult, check, format_action

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from chorus.permissions.engine import PermissionProfile
    from chorus.tools.registry import ToolDefinition, ToolRegistry

logger = logging.getLogger("chorus.llm.tool_loop")

# ---------------------------------------------------------------------------
# Permission category mapping
# ---------------------------------------------------------------------------

# Maps tool registry names to permission category names used by presets.
# The presets use categories like "file", "bash", "git" in their action
# strings, but the registry uses specific names like "create_file", "view".
_TOOL_TO_CATEGORY: dict[str, str] = {
    "create_file": "file",
    "str_replace": "file",
    "view": "file",
    "bash": "bash",
    "git_init": "git",
    "git_commit": "git",
    "git_push": "git",
    "git_branch": "git",
    "git_checkout": "git",
    "git_diff": "git",
    "git_log": "git",
    "git_merge_request": "git",
    "self_edit_system_prompt": "self_edit",
    "self_edit_docs": "self_edit",
    "self_edit_permissions": "self_edit",
    "self_edit_model": "self_edit",
}


def _build_action_string(tool_name: str, arguments: dict[str, Any]) -> str:
    """Build a permission action string for a tool call.

    Uses the category mapping so action strings match the permission
    preset patterns (e.g. ``tool:file:path/to/file``).
    """
    category = _TOOL_TO_CATEGORY.get(tool_name, tool_name)

    # Build a meaningful detail string
    if category == "file":
        detail = arguments.get("path", str(arguments))
    elif category == "bash":
        detail = arguments.get("command", str(arguments))
    elif category == "git":
        # Strip "git_" prefix for the operation name
        op = tool_name.removeprefix("git_")
        detail = f"{op} {json.dumps(arguments)}"
    elif category == "self_edit":
        # e.g. "system_prompt", "docs README.md", "permissions open", "model gpt-4o"
        sub = tool_name.removeprefix("self_edit_")
        if sub == "docs":
            detail = f"docs {arguments.get('path', '')}"
        elif sub == "system_prompt":
            detail = "system_prompt"
        elif sub == "permissions":
            detail = f"permissions {arguments.get('profile', '')}"
        elif sub == "model":
            detail = f"model {arguments.get('model', '')}"
        else:
            detail = sub
    else:
        detail = str(arguments)

    return format_action(category, detail)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ToolExecutionContext:
    """Execution context injected into tool handlers."""

    workspace: Path
    profile: PermissionProfile
    agent_name: str
    chorus_home: Path | None = None
    is_admin: bool = False
    db: Any = None


@dataclass
class ToolLoopResult:
    """Result of a complete tool loop run."""

    content: str | None
    messages: list[dict[str, Any]]
    total_usage: Usage
    iterations: int
    tool_calls_made: int


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


async def _execute_tool(
    tool: ToolDefinition,
    arguments: dict[str, Any],
    ctx: ToolExecutionContext,
) -> str:
    """Execute a tool handler, injecting context parameters as needed.

    Uses ``inspect.signature()`` to determine which context parameters
    (workspace, profile, agent_name) the handler accepts and injects
    them automatically. The LLM only provides schema-defined arguments.
    """
    sig = inspect.signature(tool.handler)
    kwargs = dict(arguments)

    # Inject context parameters if the handler accepts them
    if "workspace" in sig.parameters:
        kwargs["workspace"] = ctx.workspace
    if "profile" in sig.parameters:
        kwargs["profile"] = ctx.profile
    if "agent_name" in sig.parameters:
        kwargs["agent_name"] = ctx.agent_name
    if "chorus_home" in sig.parameters:
        kwargs["chorus_home"] = ctx.chorus_home
    if "is_admin" in sig.parameters:
        kwargs["is_admin"] = ctx.is_admin
    if "db" in sig.parameters:
        kwargs["db"] = ctx.db

    result = await tool.handler(**kwargs)

    # Normalize result to string
    if isinstance(result, str):
        return result
    if hasattr(result, "to_dict"):
        return json.dumps(result.to_dict())
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run_tool_loop(
    provider: LLMProvider,
    messages: list[dict[str, Any]],
    tools: ToolRegistry,
    ctx: ToolExecutionContext,
    system_prompt: str,
    model: str,
    max_iterations: int = 25,
    ask_callback: Callable[[str, str], Awaitable[bool]] | None = None,
    inject_queue: asyncio.Queue[dict[str, Any]] | None = None,
) -> ToolLoopResult:
    """Run the agentic tool use loop.

    Parameters
    ----------
    provider:
        The LLM provider to use (Anthropic or OpenAI).
    messages:
        Initial conversation messages (mutated in place).
    tools:
        Tool registry containing available tools.
    ctx:
        Execution context with workspace, permissions, and agent name.
    system_prompt:
        System prompt for the LLM.
    model:
        Model identifier to use.
    max_iterations:
        Maximum number of LLM calls before stopping.
    ask_callback:
        Async callback for ASK permission prompts. Called with
        ``(tool_name, arguments_str)`` and returns True if approved.

    Returns
    -------
    ToolLoopResult
        The final result including content, messages, usage, and metrics.
    """
    # Build tool schemas based on provider
    tool_defs = tools.list_all()
    if provider.provider_name == "openai":
        tool_schemas = tools_to_openai(tool_defs) if tool_defs else None
    else:
        tool_schemas = tools_to_anthropic(tool_defs) if tool_defs else None

    # Prepend system prompt as a system message if not already present
    working_messages = list(messages)
    if system_prompt and (
        not working_messages or working_messages[0].get("role") != "system"
    ):
        working_messages.insert(0, {"role": "system", "content": system_prompt})

    total_usage = Usage(input_tokens=0, output_tokens=0)
    total_tool_calls = 0

    for iteration in range(1, max_iterations + 1):
        # Drain injected messages from the queue before each LLM call
        if inject_queue is not None:
            while True:
                try:
                    injected = inject_queue.get_nowait()
                    working_messages.append(injected)
                except asyncio.QueueEmpty:
                    break

        response = await provider.chat(
            messages=working_messages,
            tools=tool_schemas,
            model=model,
        )

        total_usage = total_usage + response.usage

        # No tool calls → we're done
        if not response.tool_calls:
            return ToolLoopResult(
                content=response.content,
                messages=working_messages,
                total_usage=total_usage,
                iterations=iteration,
                tool_calls_made=total_tool_calls,
            )

        # Append assistant message with tool calls to conversation
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in response.tool_calls
            ],
        }
        working_messages.append(assistant_msg)

        # Execute each tool call
        for tc in response.tool_calls:
            result_content = await _handle_tool_call(tc, tools, ctx, ask_callback)
            working_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_content,
            })
            total_tool_calls += 1

    # Reached max iterations
    return ToolLoopResult(
        content=f"Stopped after max iterations ({max_iterations}). "
        "The task may be incomplete.",
        messages=working_messages,
        total_usage=total_usage,
        iterations=max_iterations,
        tool_calls_made=total_tool_calls,
    )


async def _handle_tool_call(
    tc: ToolCall,
    tools: ToolRegistry,
    ctx: ToolExecutionContext,
    ask_callback: Callable[[str, str], Awaitable[bool]] | None,
) -> str:
    """Handle a single tool call: permission check → execute → return result string."""
    # Look up tool
    tool = tools.get(tc.name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool: {tc.name}"})

    # Permission check
    action = _build_action_string(tc.name, tc.arguments)
    perm = check(action, ctx.profile)

    if perm is PermissionResult.DENY:
        logger.info("Permission denied for %s: %s", tc.name, action)
        return json.dumps({"error": f"Permission denied: {action}"})

    if perm is PermissionResult.ASK:
        if ask_callback is None:
            logger.info("ASK permission with no callback for %s, denying", tc.name)
            return json.dumps({"error": f"Permission requires approval (no callback): {action}"})

        args_str = json.dumps(tc.arguments)
        approved = await ask_callback(tc.name, args_str)
        if not approved:
            logger.info("User denied %s: %s", tc.name, action)
            return json.dumps({"error": f"User declined: {action}"})

    # Execute tool
    try:
        return await _execute_tool(tool, tc.arguments, ctx)
    except Exception as exc:
        logger.exception("Tool %s raised an error", tc.name)
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
