"""System prompt refinement — tailors the template prompt for new agents.

When an agent is created, a cheap LLM (Haiku / gpt-4o-mini) refines the
generic template system prompt based on the agent's name and any user-provided
description.  On any failure the template prompt is returned unchanged so
agent creation never fails because of refinement.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chorus.config import BotConfig
    from chorus.llm.providers import LLMProvider

logger = logging.getLogger("chorus.agent.prompt_refinement")

_REFINEMENT_META_PROMPT = (
    "You are configuring a new AI agent. Given the agent's name and description, "
    "refine the system prompt to be specific to this agent's role.\n\n"
    "Keep all structural elements (tool instructions, workspace rules, permission "
    "awareness, docs/ directory references) from the template. Add personality, "
    "domain expertise, and task-specific guidance based on the name and description.\n\n"
    "Output ONLY the refined system prompt, nothing else."
)


def _pick_refinement_model(
    anthropic_key: str | None,
    openai_key: str | None,
) -> tuple[str, str] | None:
    """Return ``(provider_name, model_id)`` or ``None`` if no keys."""
    if anthropic_key:
        return ("anthropic", "claude-haiku-4-5-20251001")
    if openai_key:
        return ("openai", "gpt-4o-mini")
    return None


def _create_refinement_provider(
    anthropic_key: str | None,
    openai_key: str | None,
) -> tuple[LLMProvider, str] | None:
    """Create provider + model for refinement.  ``None`` if no keys."""
    from chorus.llm.providers import AnthropicProvider, OpenAIProvider

    pick = _pick_refinement_model(anthropic_key, openai_key)
    if pick is None:
        return None

    provider_name, model = pick
    if provider_name == "anthropic":
        assert anthropic_key is not None
        return (AnthropicProvider(anthropic_key, model), model)
    assert openai_key is not None
    return (OpenAIProvider(openai_key, model), model)


async def refine_system_prompt(
    agent_name: str,
    user_description: str | None,
    template_prompt: str,
    config: BotConfig,
) -> str:
    """Refine the template system prompt using a small LLM.

    Returns the refined prompt, or the original *template_prompt*
    on any failure (missing keys, LLM error, timeout).
    """
    result = _create_refinement_provider(
        anthropic_key=config.anthropic_api_key,
        openai_key=config.openai_api_key,
    )
    if result is None:
        logger.info("No API keys available for prompt refinement — using template")
        return template_prompt

    provider, model = result

    desc = user_description or "none provided — infer from the name"
    user_content = (
        f"Agent name: {agent_name}\n"
        f"User description: {desc}\n"
        f"Template prompt:\n{template_prompt}"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _REFINEMENT_META_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        response = await asyncio.wait_for(
            provider.chat(messages=messages, model=model),
            timeout=10.0,
        )
        refined = (response.content or "").strip()
        if not refined:
            logger.warning("Refinement returned empty content — using template")
            return template_prompt
        logger.info("System prompt refined for agent %r", agent_name)
        return refined
    except Exception:
        logger.exception("Prompt refinement failed — using template")
        return template_prompt
