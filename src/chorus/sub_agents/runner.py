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
from typing import Any

from chorus.llm.providers import AnthropicProvider, LLMProvider, OpenAIProvider, Usage

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

# Provider preference order: Anthropic Haiku (cheapest), then OpenAI gpt-4o-mini.
_CHEAP_MODELS: list[tuple[str, str, str]] = [
    ("ANTHROPIC_API_KEY", "anthropic", "claude-haiku-4-5-20251001"),
    ("OPENAI_API_KEY", "openai", "gpt-4o-mini"),
]


def pick_cheap_model() -> tuple[str, str, str] | None:
    """Select the cheapest available model based on environment API keys.

    Returns ``(env_var_name, provider_name, model_id)`` or ``None`` if no
    keys are available.  Prefers Haiku over gpt-4o-mini for cost reasons.
    """
    for env_var, provider_name, model_id in _CHEAP_MODELS:
        if os.environ.get(env_var, "").strip():
            return env_var, provider_name, model_id
    return None


def _create_provider(provider_name: str, api_key: str, model: str) -> LLMProvider:
    """Instantiate a concrete provider by name."""
    if provider_name == "anthropic":
        return AnthropicProvider(api_key, model)
    return OpenAIProvider(api_key, model)


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
    empty_usage = Usage(input_tokens=0, output_tokens=0)

    # Select model
    pick = pick_cheap_model()
    if pick is None:
        return SubAgentResult(
            success=False,
            output="",
            model_used="",
            usage=empty_usage,
            error="No API keys available (ANTHROPIC_API_KEY or OPENAI_API_KEY)",
        )

    env_var, provider_name, default_model = pick
    api_key = os.environ.get(env_var, "").strip()

    # Determine model to use — override only if it matches the provider
    model = default_model
    if model_override and (
        (model_override.startswith("claude") and provider_name == "anthropic")
        or (model_override.startswith("gpt") and provider_name == "openai")
    ):
        model = model_override

    provider = _create_provider(provider_name, api_key, model)

    # Prepend system message if not already present
    working_messages = list(messages)
    if system_prompt and (
        not working_messages or working_messages[0].get("role") != "system"
    ):
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

    except TimeoutError:
        return SubAgentResult(
            success=False,
            output="",
            model_used=model,
            usage=empty_usage,
            error=f"Timeout after {timeout}s",
        )
    except Exception as exc:
        logger.exception("Sub-agent execution failed")
        return SubAgentResult(
            success=False,
            output="",
            model_used=model,
            usage=empty_usage,
            error=f"{type(exc).__name__}: {exc}",
        )
