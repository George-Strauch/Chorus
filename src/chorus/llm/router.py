"""Interjection router — decides INJECT vs NEW_THREAD for busy agents.

When a user sends a message to a channel whose main thread is already
RUNNING, a cheap LLM classifies the message as either a follow-up to
the current thread (INJECT) or an unrelated request (NEW_THREAD).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chorus.llm.providers import LLMProvider

logger = logging.getLogger("chorus.llm.router")

_ROUTER_SYSTEM_PROMPT = (
    "You are a message router. Your ONLY job is to classify whether a new user message "
    "is a follow-up to the currently running thread or a completely unrelated new request.\n\n"
    "Respond with EXACTLY one word:\n"
    "- INJECT — if the message is a follow-up, correction, "
    "clarification, or addition to the current work\n"
    "- NEW_THREAD — if the message is clearly unrelated to the current work\n\n"
    "When in doubt, respond INJECT."
)


class RouteDecision(Enum):
    """Routing decision for an interjected message."""

    INJECT = "inject"
    NEW_THREAD = "new_thread"


async def route_interjection(
    message: str,
    thread_summary: str,
    current_step: str,
    provider: LLMProvider,
    model: str,
) -> RouteDecision:
    """Ask a cheap LLM whether to inject a message or start a new thread.

    Defaults to INJECT on any error (safer — keeps conversation coherent).
    """
    user_content = (
        f"Current thread: {thread_summary}\n"
        f"Current step: {current_step}\n"
        f"New message: {message}"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        response = await provider.chat(messages=messages, model=model)
        text = (response.content or "").strip().upper()
        if "NEW_THREAD" in text:
            return RouteDecision.NEW_THREAD
        return RouteDecision.INJECT
    except Exception:
        logger.exception("Router LLM call failed — defaulting to INJECT")
        return RouteDecision.INJECT


def get_router_model(
    anthropic_key: str | None,
    openai_key: str | None,
) -> tuple[str, str]:
    """Pick the cheapest available model for routing.

    Returns ``(provider_name, model_id)``.

    Raises ``ValueError`` if no API key is available.
    """
    if anthropic_key:
        return ("anthropic", "claude-haiku-4-5-20251001")
    if openai_key:
        return ("openai", "gpt-4o-mini")
    raise ValueError("No API key available for router — need Anthropic or OpenAI key")


def create_router_provider(
    anthropic_key: str | None,
    openai_key: str | None,
) -> tuple[Any, str]:
    """Create an LLM provider instance for routing.

    Returns ``(provider_instance, model_id)``.
    """
    from chorus.llm.providers import AnthropicProvider, OpenAIProvider

    provider_name, model = get_router_model(anthropic_key, openai_key)
    if provider_name == "anthropic":
        assert anthropic_key is not None
        return (AnthropicProvider(anthropic_key, model), model)
    assert openai_key is not None
    return (OpenAIProvider(openai_key, model), model)
