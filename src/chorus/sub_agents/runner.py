"""Sub-agent runner — lightweight LLM calls without tool execution.

This module provides a simple runner for sub-agents that need cheap LLM calls
for specialized tasks like filtering, formatting, or error recovery. Sub-agents
don't have tool access — they're pure LLM calls.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chorus.llm.providers import LLMProvider, Usage

logger = logging.getLogger("chorus.sub_agents.runner")

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution."""

    success: bool
    output: str
    model_used: str
    usage: Usage
    error: str | None = None


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def pick_cheap_model() -> tuple[type[LLMProvider], str] | None:
    """Select the cheapest available model based on environment API keys.

    Returns ``(provider_class, model_id)`` or ``None`` if no keys are available.
    Prefers Haiku over gpt-4o-mini for cost reasons.
    """
    from chorus.llm.providers import AnthropicProvider, OpenAIProvider

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if anthropic_key:
        return (AnthropicProvider, "claude-haiku-4-5-20251001")
    if openai_key:
        return (OpenAIProvider, "gpt-4o-mini")

    return None


# ---------------------------------------------------------------------------
# Sub-agent runner
# ---------------------------------------------------------------------------


async def run_sub_agent(
    system_prompt: str,
    messages: list[dict[str, Any]],
    model_override: str | None = None,
    timeout: float = 15.0,
) -> SubAgentResult:
    """Run a sub-agent with a one-shot LLM call.

    This is a lightweight runner that creates a cheap LLM provider and makes
    a single chat() call. No tools are available to the sub-agent.

    Parameters
    ----------
    system_prompt:
        System prompt for the LLM.
    messages:
        List of message dicts with "role" and "content" keys.
    model_override:
        Optional specific model to use instead of auto-selection.
    timeout:
        Maximum execution time in seconds.

    Returns
    -------
    SubAgentResult
        Contains success status, output text, model used, usage stats, and
        optional error message.
    """
    from chorus.llm.providers import Usage

    # Select model
    if model_override:
        # Try to use the override model
        pick = pick_cheap_model()
        if pick is None:
            return SubAgentResult(
                success=False,
                output="",
                model_used="",
                usage=Usage(input_tokens=0, output_tokens=0),
                error="No API keys available (ANTHROPIC_API_KEY or OPENAI_API_KEY)",
            )
        provider_class, default_model = pick
        # Use the override if it matches the provider
        if model_override.startswith("claude") and provider_class.__name__ == "AnthropicProvider":
            model = model_override
        elif model_override.startswith("gpt") and provider_class.__name__ == "OpenAIProvider":
            model = model_override
        else:
            # Fallback to default if override doesn't match provider
            model = default_model
    else:
        pick = pick_cheap_model()
        if pick is None:
            return SubAgentResult(
                success=False,
                output="",
                model_used="",
                usage=Usage(input_tokens=0, output_tokens=0),
                error="No API keys available (ANTHROPIC_API_KEY or OPENAI_API_KEY)",
            )
        provider_class, model = pick

    # Get API key
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if provider_class.__name__ == "AnthropicProvider":
        if not anthropic_key:
            return SubAgentResult(
                success=False,
                output="",
                model_used=model,
                usage=Usage(input_tokens=0, output_tokens=0),
                error="ANTHROPIC_API_KEY not available",
            )
        provider = provider_class(anthropic_key, model)
    else:
        if not openai_key:
            return SubAgentResult(
                success=False,
                output="",
                model_used=model,
                usage=Usage(input_tokens=0, output_tokens=0),
                error="OPENAI_API_KEY not available",
            )
        provider = provider_class(openai_key, model)

    # Prepend system message if not already present
    working_messages = list(messages)
    if system_prompt and (not working_messages or working_messages[0].get("role") != "system"):
        working_messages.insert(0, {"role": "system", "content": system_prompt})

    # Execute with timeout
    try:
        response = await asyncio.wait_for(
            provider.chat(messages=working_messages, tools=None, model=model),
            timeout=timeout,
        )

        output = (response.content or "").strip()
        if not output:
            return SubAgentResult(
                success=False,
                output="",
                model_used=response.model,
                usage=response.usage,
                error="LLM returned empty content",
            )

        return SubAgentResult(
            success=True,
            output=output,
            model_used=response.model,
            usage=response.usage,
        )

    except asyncio.TimeoutError:
        return SubAgentResult(
            success=False,
            output="",
            model_used=model,
            usage=Usage(input_tokens=0, output_tokens=0),
            error=f"Timeout after {timeout}s",
        )
    except Exception as exc:
        logger.exception("Sub-agent execution failed")
        return SubAgentResult(
            success=False,
            output="",
            model_used=model,
            usage=Usage(input_tokens=0, output_tokens=0),
            error=f"{type(exc).__name__}: {exc}",
        )
