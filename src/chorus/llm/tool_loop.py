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
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from chorus.agent.context import MAX_INPUT_TOKENS, estimate_message_tokens
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
    "self_edit_web_search": "self_edit",
    "list_models": "info",
    "web_search": "web_search",
    "claude_code": "claude_code",
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
        elif sub == "web_search":
            detail = f"web_search {arguments.get('enabled', '')}"
        else:
            detail = sub
    elif category == "claude_code":
        detail = arguments.get("task", "")[:100]
    elif category == "web_search":
        detail = "enabled"
    else:
        detail = str(arguments)

    return format_action(category, detail)


# ---------------------------------------------------------------------------
# Mid-loop message truncation
# ---------------------------------------------------------------------------


def _truncate_tool_loop_messages(
    messages: list[dict[str, Any]], budget: int
) -> list[dict[str, Any]]:
    """Truncate tool-loop messages to fit within a token budget.

    Groups assistant+tool_result messages into atomic blocks so that a
    tool_call is never separated from its results (which would cause an
    API error).  Keeps the most recent blocks within budget.
    """
    if not messages:
        return messages

    # 1. Separate system messages from conversation messages
    system_msgs: list[dict[str, Any]] = []
    conv_msgs: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            system_msgs.append(msg)
        else:
            conv_msgs.append(msg)

    # 2. Group into atomic blocks
    #    An assistant msg with tool_calls + all following role:"tool" msgs = one block
    blocks: list[list[dict[str, Any]]] = []
    i = 0
    while i < len(conv_msgs):
        msg = conv_msgs[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Start an atomic block: assistant + following tool results
            block = [msg]
            i += 1
            while i < len(conv_msgs) and conv_msgs[i].get("role") == "tool":
                block.append(conv_msgs[i])
                i += 1
            blocks.append(block)
        else:
            blocks.append([msg])
            i += 1

    # 3. Calculate system token overhead
    system_tokens = sum(estimate_message_tokens(m) for m in system_msgs)
    remaining_budget = budget - system_tokens

    if remaining_budget <= 0:
        # System messages alone exceed budget — return system + last block
        return system_msgs + (blocks[-1] if blocks else [])

    # 4. Walk backwards through blocks, accumulating tokens until budget hit
    kept: list[list[dict[str, Any]]] = []
    total = 0
    for block in reversed(blocks):
        block_tokens = sum(estimate_message_tokens(m) for m in block)
        if total + block_tokens > remaining_budget:
            break
        kept.append(block)
        total += block_tokens

    kept.reverse()

    # 5. Reassemble
    result = list(system_msgs)
    for block in kept:
        result.extend(block)
    return result


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
    host_execution: bool = False


@dataclass
class ToolLoopResult:
    """Result of a complete tool loop run."""

    content: str | None
    messages: list[dict[str, Any]]
    total_usage: Usage
    iterations: int
    tool_calls_made: int


class ToolLoopEventType(Enum):
    """Lifecycle events emitted during the tool loop."""

    LLM_CALL_START = "llm_call_start"
    LLM_CALL_COMPLETE = "llm_call_complete"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_COMPLETE = "tool_call_complete"
    LOOP_COMPLETE = "loop_complete"


@dataclass
class ToolLoopEvent:
    """Event payload for tool loop lifecycle callbacks."""

    type: ToolLoopEventType
    iteration: int
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    usage_delta: Usage | None = None
    total_usage: Usage | None = None
    tool_calls_made: int = 0
    tools_used: list[str] = field(default_factory=list)
    content_preview: str | None = None


async def _fire_event(
    on_event: Callable[[ToolLoopEvent], Awaitable[None]] | None,
    event: ToolLoopEvent,
) -> None:
    """Fire an event callback, swallowing any errors."""
    if on_event is None:
        return
    try:
        await on_event(event)
    except Exception:
        logger.warning("on_event callback error for %s", event.type, exc_info=True)


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

    # Inject context parameters if the handler accepts them AND the LLM
    # didn't already provide a value for that name.  This prevents
    # collisions like edit_permissions' ``profile`` (a preset name string
    # from the LLM) being overwritten by ``ctx.profile`` (a
    # PermissionProfile object).
    if "workspace" in sig.parameters and "workspace" not in arguments:
        kwargs["workspace"] = ctx.workspace
    if "profile" in sig.parameters and "profile" not in arguments:
        kwargs["profile"] = ctx.profile
    if "agent_name" in sig.parameters and "agent_name" not in arguments:
        kwargs["agent_name"] = ctx.agent_name
    if "chorus_home" in sig.parameters and "chorus_home" not in arguments:
        kwargs["chorus_home"] = ctx.chorus_home
    if "is_admin" in sig.parameters and "is_admin" not in arguments:
        kwargs["is_admin"] = ctx.is_admin
    if "db" in sig.parameters and "db" not in arguments:
        kwargs["db"] = ctx.db
    if "host_execution" in sig.parameters and "host_execution" not in arguments:
        kwargs["host_execution"] = ctx.host_execution

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
    on_event: Callable[[ToolLoopEvent], Awaitable[None]] | None = None,
    web_search_enabled: bool = False,
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

    # Inject Anthropic web search tool if enabled
    if web_search_enabled and provider.provider_name != "openai":
        # One-time permission pre-check for web search
        action = format_action("web_search", "enabled")
        perm = check(action, ctx.profile)
        if perm is PermissionResult.DENY:
            web_search_enabled = False
            logger.info("Web search denied by permission profile")
        elif perm is PermissionResult.ASK:
            if ask_callback is None:
                web_search_enabled = False
                logger.info("Web search ASK with no callback, disabling")
            else:
                approved = await ask_callback("web_search", '{"enabled": true}')
                if not approved:
                    web_search_enabled = False
                    logger.info("User denied web search")

        if web_search_enabled:
            web_search_spec = {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }
            if tool_schemas is None:
                tool_schemas = [web_search_spec]
            else:
                tool_schemas.append(web_search_spec)

    # Prepend system prompt as a system message if not already present
    working_messages = list(messages)
    if system_prompt and (not working_messages or working_messages[0].get("role") != "system"):
        working_messages.insert(0, {"role": "system", "content": system_prompt})

    total_usage = Usage(input_tokens=0, output_tokens=0)
    total_tool_calls = 0
    tools_used: list[str] = []

    for iteration in range(1, max_iterations + 1):
        # Drain injected messages from the queue before each LLM call
        if inject_queue is not None:
            while True:
                try:
                    injected = inject_queue.get_nowait()
                    working_messages.append(injected)
                except asyncio.QueueEmpty:
                    break

        await _fire_event(
            on_event,
            ToolLoopEvent(
                type=ToolLoopEventType.LLM_CALL_START,
                iteration=iteration,
                tool_calls_made=total_tool_calls,
                tools_used=list(tools_used),
                total_usage=total_usage,
            ),
        )

        # Truncate to stay within the hard token cap
        working_messages = _truncate_tool_loop_messages(
            working_messages, MAX_INPUT_TOKENS
        )

        response = await provider.chat(
            messages=working_messages,
            tools=tool_schemas,
            model=model,
        )

        total_usage = total_usage + response.usage

        await _fire_event(
            on_event,
            ToolLoopEvent(
                type=ToolLoopEventType.LLM_CALL_COMPLETE,
                iteration=iteration,
                usage_delta=response.usage,
                total_usage=total_usage,
                tool_calls_made=total_tool_calls,
                tools_used=list(tools_used),
            ),
        )

        # No tool calls — check for server-side tool results (web search)
        if not response.tool_calls:
            if response._raw_content is not None:
                # Server-side tool results (web search) — append and continue
                # so the LLM can process the search results
                working_messages.append({
                    "role": "assistant",
                    "content": response.content or "",
                    "_anthropic_content": response._raw_content,
                })
                continue

            await _fire_event(
                on_event,
                ToolLoopEvent(
                    type=ToolLoopEventType.LOOP_COMPLETE,
                    iteration=iteration,
                    total_usage=total_usage,
                    tool_calls_made=total_tool_calls,
                    tools_used=list(tools_used),
                    content_preview=(response.content or "")[:200],
                ),
            )
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
        if response._raw_content is not None:
            assistant_msg["_anthropic_content"] = response._raw_content
        working_messages.append(assistant_msg)

        # Execute each tool call
        for tc in response.tool_calls:
            await _fire_event(
                on_event,
                ToolLoopEvent(
                    type=ToolLoopEventType.TOOL_CALL_START,
                    iteration=iteration,
                    tool_name=tc.name,
                    tool_arguments=tc.arguments,
                    tool_calls_made=total_tool_calls,
                    tools_used=list(tools_used),
                    total_usage=total_usage,
                ),
            )

            result_content = await _handle_tool_call(tc, tools, ctx, ask_callback)
            working_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_content,
                }
            )
            total_tool_calls += 1
            if tc.name not in tools_used:
                tools_used.append(tc.name)

            await _fire_event(
                on_event,
                ToolLoopEvent(
                    type=ToolLoopEventType.TOOL_CALL_COMPLETE,
                    iteration=iteration,
                    tool_name=tc.name,
                    tool_calls_made=total_tool_calls,
                    tools_used=list(tools_used),
                    total_usage=total_usage,
                ),
            )

    # Reached max iterations
    await _fire_event(
        on_event,
        ToolLoopEvent(
            type=ToolLoopEventType.LOOP_COMPLETE,
            iteration=max_iterations,
            total_usage=total_usage,
            tool_calls_made=total_tool_calls,
            tools_used=list(tools_used),
        ),
    )
    return ToolLoopResult(
        content=f"Stopped after max iterations ({max_iterations}). The task may be incomplete.",
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
