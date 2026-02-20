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

# Parameters that are injected by _execute_tool, not provided by the LLM.
_CONTEXT_INJECTED_PARAMS = frozenset({
    "workspace", "profile", "agent_name", "chorus_home",
    "is_admin", "db", "host_execution", "scope_path",
    "process_manager", "branch_id", "on_tool_progress",
    "hook_dispatcher", "bot",
})

logger = logging.getLogger("chorus.llm.tool_loop")

# ---------------------------------------------------------------------------
# Permission category mapping
# ---------------------------------------------------------------------------

# Maps tool registry names to permission category names used by presets.
# The presets use categories like "file", "bash", "git" in their action
# strings, but the registry uses specific names like "create_file", "view".
_TOOL_TO_CATEGORY: dict[str, str] = {
    "create_file": "file",
    "append_file": "file",
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
    "git_status": "git",
    "self_edit_system_prompt": "self_edit",
    "self_edit_docs": "self_edit",
    "self_edit_permissions": "self_edit",
    "self_edit_model": "self_edit",
    "self_edit_web_search": "self_edit",
    "list_models": "info",
    "web_search": "web_search",
    "claude_code": "claude_code",
    "run_concurrent": "run_concurrent",
    "run_background": "run_background",
    "add_process_hooks": "add_process_hooks",
    "send_to_agent": "agent_comm",
    "read_agent_docs": "agent_comm",
    "list_agents": "agent_comm",
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
    elif category in ("run_concurrent", "run_background"):
        detail = arguments.get("command", str(arguments))
    elif category == "add_process_hooks":
        detail = str(arguments.get("pid", ""))
    elif category == "agent_comm":
        if tool_name == "send_to_agent":
            detail = f"send {arguments.get('target_agent', '')}"
        elif tool_name == "read_agent_docs":
            detail = f"read_docs {arguments.get('target_agent', '')}"
        else:
            detail = "list"
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
    scope_path: Any = None  # Path | None — auto-detect host credential resolution
    process_manager: Any = None
    hook_dispatcher: Any = None
    branch_id: int | None = None
    on_tool_progress: Any = None  # Callable[[dict], Any] | None
    bot: Any = None  # ChorusBot — injected for inter-agent communication


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


def _is_error_result(content: str) -> bool:
    """Check if a tool result string represents an error (JSON with "error" key)."""
    try:
        data = json.loads(content)
        return isinstance(data, dict) and "error" in data
    except (json.JSONDecodeError, TypeError):
        return False


def _validate_tool_arguments(
    tool: ToolDefinition,
    arguments: dict[str, Any],
) -> str | None:
    """Pre-validate that all required schema arguments are present.

    Returns ``None`` if valid, or a descriptive error message listing
    the expected parameters when required arguments are missing.
    Context-injected parameters (workspace, profile, etc.) are excluded
    from the check since the LLM is not expected to provide them.
    """
    schema = tool.parameters
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    # Filter out context-injected params — the LLM shouldn't provide these
    required_from_llm = [r for r in required if r not in _CONTEXT_INJECTED_PARAMS]

    missing = [r for r in required_from_llm if r not in arguments]
    if not missing:
        return None

    # Build a helpful error listing all expected parameters
    lines = [f"Missing required argument(s): {', '.join(repr(m) for m in missing)}."]
    lines.append(f"Tool '{tool.name}' expects:")
    for param_name in required_from_llm:
        prop = properties.get(param_name, {})
        ptype = prop.get("type", "any")
        desc = prop.get("description", "")
        req_marker = "(required)"
        desc_part = f" — {desc}" if desc else ""
        lines.append(f"  - {param_name}: {ptype} {req_marker}{desc_part}")

    return "\n".join(lines)


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
    if "scope_path" in sig.parameters and "scope_path" not in arguments:
        kwargs["scope_path"] = ctx.scope_path
    if "process_manager" in sig.parameters and "process_manager" not in arguments:
        kwargs["process_manager"] = ctx.process_manager
    if "hook_dispatcher" in sig.parameters and "hook_dispatcher" not in arguments:
        kwargs["hook_dispatcher"] = ctx.hook_dispatcher
    if "branch_id" in sig.parameters and "branch_id" not in arguments:
        kwargs["branch_id"] = ctx.branch_id
    if "on_tool_progress" in sig.parameters and "on_tool_progress" not in arguments:
        kwargs["on_tool_progress"] = ctx.on_tool_progress
    if "bot" in sig.parameters and "bot" not in arguments:
        kwargs["bot"] = ctx.bot

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
    consecutive_errors = 0
    max_consecutive_errors = 5

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

        # Guard: if stop_reason is "max_tokens", tool call arguments may be
        # truncated (the API cut off mid-JSON).  Discard the tool calls and
        # tell the LLM so it can retry with a shorter response.
        if response.stop_reason == "max_tokens" and response.tool_calls:
            logger.warning(
                "Response hit max_tokens with %d tool call(s) — "
                "arguments likely truncated, discarding",
                len(response.tool_calls),
            )
            truncation_msg = (
                "Your response was cut off (max_tokens reached) while generating "
                "tool call arguments. The tool call was NOT executed because the "
                "arguments were incomplete. To write large files, use append_file "
                "in multiple calls to build the content incrementally instead of "
                "trying to send it all at once via create_file."
            )
            working_messages.append({
                "role": "assistant",
                "content": response.content or "",
            })
            # Feed the error back as a synthetic user message so the LLM can adapt
            working_messages.append({
                "role": "user",
                "content": f"[system: {truncation_msg}]",
            })
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                msg = (
                    f"Stopped after {consecutive_errors} consecutive tool errors. "
                    f"The response keeps hitting the output token limit. "
                    f"Total tool calls: {total_tool_calls}, iterations: {iteration}."
                )
                logger.warning("Circuit breaker tripped (max_tokens): %s", msg)
                await _fire_event(
                    on_event,
                    ToolLoopEvent(
                        type=ToolLoopEventType.LOOP_COMPLETE,
                        iteration=iteration,
                        total_usage=total_usage,
                        tool_calls_made=total_tool_calls,
                        tools_used=list(tools_used),
                        content_preview=msg[:200],
                    ),
                )
                return ToolLoopResult(
                    content=msg,
                    messages=working_messages,
                    total_usage=total_usage,
                    iterations=iteration,
                    tool_calls_made=total_tool_calls,
                )
            continue

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

        # Execute tool calls — parallel when safe, sequential otherwise
        use_parallel = len(response.tool_calls) > 1 and _can_run_parallel(
            response.tool_calls, tools, ctx
        )

        if use_parallel:
            results = await _run_parallel(
                response.tool_calls, tools, ctx, ask_callback,
                on_event, iteration, total_tool_calls, list(tools_used), total_usage,
            )

            batch_errors = 0
            for tc, result in zip(response.tool_calls, results, strict=True):
                if isinstance(result, BaseException):
                    result_content = json.dumps(
                        {"error": f"{type(result).__name__}: {result}"}
                    )
                    tool_cost = 0.0
                    batch_errors += 1
                else:
                    result_content, tool_cost, _ = result
                    # Check if the result itself is an error
                    if _is_error_result(result_content):
                        batch_errors += 1
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
                if tool_cost > 0:
                    total_usage = total_usage + Usage(
                        input_tokens=0, output_tokens=0, cost_usd=tool_cost
                    )

            # Circuit breaker: if ALL calls in the batch failed, count batch size
            if batch_errors == len(response.tool_calls):
                consecutive_errors += batch_errors
            else:
                consecutive_errors = 0
        else:
            # Sequential execution
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

                result_content, tool_cost = await _handle_tool_call(
                    tc, tools, ctx, ask_callback
                )
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
                if tool_cost > 0:
                    total_usage = total_usage + Usage(
                        input_tokens=0, output_tokens=0, cost_usd=tool_cost
                    )

                # Circuit breaker tracking
                if _is_error_result(result_content):
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0

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

        # Circuit breaker: stop if too many consecutive errors
        if consecutive_errors >= max_consecutive_errors:
            msg = (
                f"Stopped after {consecutive_errors} consecutive tool errors. "
                f"The LLM may be sending invalid arguments repeatedly. "
                f"Total tool calls: {total_tool_calls}, iterations: {iteration}."
            )
            logger.warning("Circuit breaker tripped: %s", msg)
            await _fire_event(
                on_event,
                ToolLoopEvent(
                    type=ToolLoopEventType.LOOP_COMPLETE,
                    iteration=iteration,
                    total_usage=total_usage,
                    tool_calls_made=total_tool_calls,
                    tools_used=list(tools_used),
                    content_preview=msg[:200],
                ),
            )
            return ToolLoopResult(
                content=msg,
                messages=working_messages,
                total_usage=total_usage,
                iterations=iteration,
                tool_calls_made=total_tool_calls,
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


async def _run_parallel(
    tool_calls: list[ToolCall],
    tools: ToolRegistry,
    ctx: ToolExecutionContext,
    ask_callback: Callable[[str, str], Awaitable[bool]] | None,
    on_event: Callable[[ToolLoopEvent], Awaitable[None]] | None,
    iteration: int,
    tool_calls_made: int,
    tools_used: list[str],
    total_usage: Usage,
) -> list[tuple[str, float, str] | BaseException]:
    """Run multiple tool calls concurrently via asyncio.gather."""

    async def _run_one(tc: ToolCall) -> tuple[str, float, str]:
        await _fire_event(
            on_event,
            ToolLoopEvent(
                type=ToolLoopEventType.TOOL_CALL_START,
                iteration=iteration,
                tool_name=tc.name,
                tool_arguments=tc.arguments,
                tool_calls_made=tool_calls_made,
                tools_used=list(tools_used),
                total_usage=total_usage,
            ),
        )
        result_content, cost = await _handle_tool_call(tc, tools, ctx, ask_callback)
        await _fire_event(
            on_event,
            ToolLoopEvent(
                type=ToolLoopEventType.TOOL_CALL_COMPLETE,
                iteration=iteration,
                tool_name=tc.name,
                tool_calls_made=tool_calls_made,
                tools_used=list(tools_used),
                total_usage=total_usage,
            ),
        )
        return result_content, cost, tc.name

    return await asyncio.gather(
        *[_run_one(tc) for tc in tool_calls],
        return_exceptions=True,
    )


def _can_run_parallel(
    tool_calls: list[ToolCall],
    tools: ToolRegistry,
    ctx: ToolExecutionContext,
) -> bool:
    """Check whether a batch of tool calls can safely run in parallel.

    Returns False if any tool call would require ASK permission (we can't
    show multiple permission prompts simultaneously in Discord).
    """
    for tc in tool_calls:
        tool = tools.get(tc.name)
        if tool is None:
            continue  # Unknown tools will error; that's fine in parallel
        action = _build_action_string(tc.name, tc.arguments)
        perm = check(action, ctx.profile)
        if perm is PermissionResult.ASK:
            return False
    return True


async def _handle_tool_call(
    tc: ToolCall,
    tools: ToolRegistry,
    ctx: ToolExecutionContext,
    ask_callback: Callable[[str, str], Awaitable[bool]] | None,
) -> tuple[str, float]:
    """Handle a single tool call: permission check → execute → return (result, cost_usd)."""
    # Look up tool
    tool = tools.get(tc.name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool: {tc.name}"}), 0.0

    # Permission check
    action = _build_action_string(tc.name, tc.arguments)
    perm = check(action, ctx.profile)

    if perm is PermissionResult.DENY:
        logger.info("Permission denied for %s: %s", tc.name, action)
        return json.dumps({"error": f"Permission denied: {action}"}), 0.0

    if perm is PermissionResult.ASK:
        if ask_callback is None:
            logger.info("ASK permission with no callback for %s, denying", tc.name)
            return (
                json.dumps({"error": f"Permission requires approval (no callback): {action}"}),
                0.0,
            )

        args_str = json.dumps(tc.arguments)
        approved = await ask_callback(tc.name, args_str)
        if not approved:
            logger.info("User denied %s: %s", tc.name, action)
            return json.dumps({"error": f"User declined: {action}"}), 0.0

    # Pre-validate required arguments
    validation_error = _validate_tool_arguments(tool, tc.arguments)
    if validation_error is not None:
        logger.warning("Argument validation failed for %s: %s", tc.name, validation_error)
        return json.dumps({"error": validation_error}), 0.0

    # Execute tool
    try:
        result_str = await _execute_tool(tool, tc.arguments, ctx)
    except TypeError as exc:
        # Provide detailed info about provided vs expected arguments
        sig = inspect.signature(tool.handler)
        expected = list(sig.parameters.keys())
        provided = list(tc.arguments.keys())
        msg = (
            f"TypeError calling '{tc.name}': {exc}\n"
            f"  Provided arguments: {provided}\n"
            f"  Expected parameters: {expected}"
        )
        logger.exception("Tool %s TypeError", tc.name)
        return json.dumps({"error": msg}), 0.0
    except Exception as exc:
        logger.exception("Tool %s raised an error", tc.name)
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}), 0.0

    # Extract cost from claude_code results
    cost_usd = 0.0
    if tc.name == "claude_code":
        try:
            result_data = json.loads(result_str)
            cost_usd = result_data.get("cost_usd") or 0.0
        except (json.JSONDecodeError, TypeError):
            pass

    return result_str, cost_usd
