"""CallbackBuilder — translate NL instructions into structured ProcessCallbacks."""

from __future__ import annotations

import json
import logging
from typing import Any

from chorus.process.models import (
    CallbackAction,
    ExitFilter,
    HookTrigger,
    ProcessCallback,
    TriggerType,
)
from chorus.sub_agents.runner import run_sub_agent

logger = logging.getLogger("chorus.process.callback_builder")

_SYSTEM_PROMPT = """\
You are a callback configuration assistant. Given a user's natural language \
instructions about what should happen with a running process, you produce a \
JSON array of callback objects.

Each callback has:
- trigger: {"type": "on_exit"|"on_output_match"|"on_timeout", \
  "exit_filter": "any"|"success"|"failure", "pattern": "regex", \
  "timeout_seconds": number}
- action: "stop_process"|"stop_branch"|"inject_context"|"spawn_branch"|"notify_channel"
- context_message: string (message to include when action fires)
- output_delay_seconds: number (seconds to wait after match before firing, default 2.0)
- max_fires: integer (how many times this callback can fire, default 1)

Common patterns:
- "notify me when it finishes" → on_exit(any) → notify_channel
- "if it fails, fix it" → on_exit(failure) → spawn_branch with context "Fix it"
- "stop if you see an error" → on_output_match("error|Error|ERROR") → stop_process
- "if compilation succeeds, continue" → on_exit(success) → inject_context
- "kill it after 5 minutes" → on_timeout(300) → stop_process

Respond ONLY with a JSON array. No explanation.
"""

_DEFAULT_CALLBACK = ProcessCallback(
    trigger=HookTrigger(type=TriggerType.ON_EXIT, exit_filter=ExitFilter.ANY),
    action=CallbackAction.NOTIFY_CHANNEL,
    context_message="Process completed",
    max_fires=1,
)


async def build_callbacks_from_instructions(
    instructions: str,
    command: str = "",
    default_output_delay: float = 2.0,
) -> list[ProcessCallback]:
    """Translate NL instructions into structured ProcessCallbacks.

    Uses a cheap Haiku sub-agent call. Falls back to a default
    on_exit(any) → spawn_branch callback on failure.

    Parameters
    ----------
    instructions:
        Natural language description of desired behavior.
    command:
        The command being run (for context).
    default_output_delay:
        Default output delay for on_output_match callbacks.
    """
    if not instructions.strip():
        return [_DEFAULT_CALLBACK]

    user_message = f"Command: `{command}`\nInstructions: {instructions}"

    try:
        result = await run_sub_agent(
            system_prompt=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            timeout=10.0,
        )

        if not result.success:
            logger.warning("Callback builder sub-agent failed: %s", result.error)
            return [_DEFAULT_CALLBACK]

        callbacks = _parse_callbacks(result.output, default_output_delay)
        if not callbacks:
            logger.warning("Callback builder returned empty result")
            return [_DEFAULT_CALLBACK]

        return callbacks

    except Exception:
        logger.exception("Callback builder failed")
        return [_DEFAULT_CALLBACK]


def _parse_callbacks(
    raw_output: str,
    default_output_delay: float,
) -> list[ProcessCallback]:
    """Parse the LLM's JSON output into ProcessCallback objects."""
    # Strip markdown code fences if present
    text = raw_output.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse callback builder output as JSON")
        return []

    if not isinstance(data, list):
        data = [data]

    callbacks: list[ProcessCallback] = []
    for item in data:
        try:
            cb = _parse_single_callback(item, default_output_delay)
            callbacks.append(cb)
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping invalid callback: %s — %s", item, exc)
            continue

    return callbacks


def _parse_single_callback(
    item: dict[str, Any],
    default_output_delay: float,
) -> ProcessCallback:
    """Parse a single callback dict from the LLM output."""
    trigger_data = item.get("trigger", {})
    trigger_type = TriggerType(trigger_data.get("type", "on_exit"))

    exit_filter = ExitFilter(trigger_data.get("exit_filter", "any"))
    pattern = trigger_data.get("pattern")
    timeout_seconds = trigger_data.get("timeout_seconds")

    trigger = HookTrigger(
        type=trigger_type,
        exit_filter=exit_filter,
        pattern=pattern,
        timeout_seconds=timeout_seconds,
    )

    action = CallbackAction(item.get("action", "spawn_branch"))
    context_message = item.get("context_message", "")

    # Use default delay for output match if not specified
    output_delay = item.get("output_delay_seconds")
    if output_delay is None and trigger_type == TriggerType.ON_OUTPUT_MATCH:
        output_delay = default_output_delay
    elif output_delay is None:
        output_delay = 0.0

    max_fires = item.get("max_fires", 1)

    return ProcessCallback(
        trigger=trigger,
        action=action,
        context_message=context_message,
        output_delay_seconds=output_delay,
        max_fires=max_fires,
    )
