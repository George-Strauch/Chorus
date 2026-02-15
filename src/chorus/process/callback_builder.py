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
You are a callback configuration assistant. Given natural language instructions \
about what should happen with a running process, produce a JSON array of \
callback objects.

## Schema

Each callback object has these fields:

```
{
  "trigger": {
    "type": "on_exit" | "on_output_match" | "on_timeout",
    "exit_filter": "any" | "success" | "failure",  // only for on_exit
    "pattern": "regex",                             // only for on_output_match
    "timeout_seconds": number                       // only for on_timeout
  },
  "action": "stop_process" | "stop_branch" | "inject_context" | "spawn_branch" | "notify_channel",
  "context_message": "string — passed to the action handler as context",
  "output_delay_seconds": number,  // wait before firing on_output_match (default 2.0)
  "max_fires": integer,            // how many times this callback can fire (default 1)
  "min_message_interval": number   // rate-limit seconds between notify_channel fires (default 180)
}
```

## Actions explained

- **stop_process**: Kill the monitored process.
- **stop_branch**: Kill the LLM execution branch that started this process.
- **inject_context**: Send a message into the current branch's conversation (the LLM will see it).
- **spawn_branch**: Start a NEW autonomous LLM branch with context_message as instructions. \
The new branch can read output, run commands, fix issues, or continue work. This is the \
primary way to chain autonomous reactions.
- **notify_channel**: Post a notification to the Discord channel \
(informational only, no LLM action).

## Examples

1. "notify me when it finishes"
```json
[{"trigger": {"type": "on_exit", "exit_filter": "any"},
  "action": "notify_channel",
  "context_message": "Process finished"}]
```

2. "if it fails, diagnose and fix the issue"
```json
[{"trigger": {"type": "on_exit", "exit_filter": "failure"},
  "action": "spawn_branch",
  "context_message": "The process failed. Diagnose and fix it."}]
```

3. "stop the process if errors appear in the output"
```json
[{"trigger": {"type": "on_output_match",
              "pattern": "[Ee]rror|FATAL|panic"},
  "action": "stop_process",
  "context_message": "Error detected in output"}]
```

4. "when it succeeds, tell me and run the next step"
```json
[{"trigger": {"type": "on_exit", "exit_filter": "success"},
  "action": "notify_channel",
  "context_message": "Step completed successfully"},
 {"trigger": {"type": "on_exit", "exit_filter": "success"},
  "action": "spawn_branch",
  "context_message": "Previous step succeeded. Proceed."}]
```

5. "kill it after 10 minutes"
```json
[{"trigger": {"type": "on_timeout", "timeout_seconds": 600},
  "action": "stop_process",
  "context_message": "Process timed out"}]
```

6. "on success, run the same script again with incremented params"
```json
[{"trigger": {"type": "on_exit", "exit_filter": "success"},
  "action": "spawn_branch",
  "context_message": "Previous run succeeded. Run next iteration."}]
```

7. "if it fails, notify me; if it succeeds, continue working"
```json
[{"trigger": {"type": "on_exit", "exit_filter": "failure"},
  "action": "notify_channel",
  "context_message": "Process failed"},
 {"trigger": {"type": "on_exit", "exit_filter": "success"},
  "action": "spawn_branch",
  "context_message": "Succeeded. Continue with the next task."}]
```

8. "watch for 'ready' in output, then tell this branch to proceed"
```json
[{"trigger": {"type": "on_output_match",
              "pattern": "ready|Ready|READY"},
  "action": "inject_context",
  "context_message": "Server is ready. Start sending requests.",
  "output_delay_seconds": 1.0}]
```

## Guidelines

- The context_message for spawn_branch should be a CLEAR INSTRUCTION telling the new \
branch what to do. It will be the opening message in a new autonomous LLM conversation.
- Include enough detail in context_message for the spawned branch to act independently.
- You can combine multiple callbacks (e.g. notify on failure AND spawn_branch on success).
- For chaining/recursion patterns (run step N, then step N+1), use spawn_branch with \
context_message that describes what the next step should do.
- Use inject_context when you want the CURRENT branch to react (it must still be running).
- Use spawn_branch when you want a NEW branch to handle something autonomously.
- max_fires: 0 means unlimited (fire every time). max_fires: N (N>0) means fire at most N times.
- For on_output_match hooks, default to max_fires: 0 (unlimited) unless the user specifies a limit.
- For on_exit and on_timeout hooks, default to max_fires: 1 unless the user specifies otherwise.
- Use max_fires > 1 for repeating triggers (e.g. restart on every failure, up to 3 times).
- For notify_channel with on_output_match, min_message_interval rate-limits notifications (default 180s / 3 min).

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

    max_fires = item.get("max_fires")
    if max_fires is None:
        max_fires = 0 if trigger_type == TriggerType.ON_OUTPUT_MATCH else 1

    min_message_interval = item.get("min_message_interval", 180.0)

    return ProcessCallback(
        trigger=trigger,
        action=action,
        context_message=context_message,
        output_delay_seconds=output_delay,
        max_fires=max_fires,
        min_message_interval=min_message_interval,
    )
