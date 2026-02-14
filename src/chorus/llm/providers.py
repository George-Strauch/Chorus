"""Multi-provider LLM client — Anthropic and OpenAI.

Normalizes both providers into a unified ``LLMResponse`` dataclass so the
tool loop is provider-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from chorus.tools.registry import ToolDefinition


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    """Token usage for a single LLM call."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
        )


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""

    content: str | None
    tool_calls: list[ToolCall]
    stop_reason: str
    usage: Usage
    model: str
    _raw_content: list[dict[str, Any]] | None = None


@dataclass
class LLMChunk:
    """A single chunk from a streaming LLM response."""

    delta: str
    finish_reason: str | None = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """Interface that all LLM providers must implement."""

    @property
    def provider_name(self) -> str: ...

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Tool schema translation
# ---------------------------------------------------------------------------


def tools_to_anthropic(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert tool definitions to Anthropic's tool schema format."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


def tools_to_openai(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert tool definitions to OpenAI's function calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# Message format translation helpers
# ---------------------------------------------------------------------------


def _messages_to_anthropic(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Extract system prompt and translate messages to Anthropic format.

    Returns ``(system_prompt, translated_messages)``.

    Neutral format conversions:
    - System messages → extracted into a single system prompt string
    - Assistant messages with ``tool_calls`` → content blocks with tool_use
    - Tool result messages → user messages with tool_result blocks
    """
    system_parts: list[str] = []
    translated: list[dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]

        if role == "system":
            system_parts.append(msg["content"])

        elif role == "assistant" and msg.get("_anthropic_content"):
            # Preserve raw Anthropic content blocks (e.g. web search results)
            translated.append({"role": "assistant", "content": msg["_anthropic_content"]})

        elif role == "assistant" and msg.get("tool_calls"):
            # Build content blocks: text + tool_use blocks
            content_blocks: list[dict[str, Any]] = []
            if msg.get("content"):
                content_blocks.append({"type": "text", "text": msg["content"]})
            for tc in msg["tool_calls"]:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["arguments"],
                })
            translated.append({"role": "assistant", "content": content_blocks})

        elif role == "tool":
            # Tool results become user messages with tool_result blocks
            translated.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg["tool_call_id"],
                        "content": msg["content"],
                    }
                ],
            })

        else:
            # Regular user/assistant messages pass through
            translated.append({"role": role, "content": msg["content"]})

    return "\n\n".join(system_parts), translated


def _messages_to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate messages to OpenAI format.

    Neutral format conversions:
    - System messages pass through
    - Assistant messages with ``tool_calls`` → OpenAI tool_calls format
    - Tool result messages pass through (OpenAI uses role "tool" natively)
    """
    translated: list[dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]

        if role == "assistant" and msg.get("tool_calls"):
            oai_tool_calls = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in msg["tool_calls"]
            ]
            translated.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": oai_tool_calls,
            })

        elif role == "tool":
            translated.append({
                "role": "tool",
                "tool_call_id": msg["tool_call_id"],
                "content": msg["content"],
            })

        else:
            # system, user, plain assistant — pass through
            translated.append({"role": role, "content": msg["content"]})

    return translated


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


class AnthropicProvider:
    """LLM provider using the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        default_model: str,
        *,
        _client: Any = None,
    ) -> None:
        self._default_model = default_model
        if _client is not None:
            self._client = _client
        else:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        system_prompt, translated = _messages_to_anthropic(messages)

        kwargs: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": translated,
            "max_tokens": 4096,
        }
        if system_prompt:
            kwargs["system"] = [
                {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}},
            ]
        if tools:
            tools_copy = [dict(t) for t in tools]  # Don't mutate caller's list
            tools_copy[-1] = {**tools_copy[-1], "cache_control": {"type": "ephemeral"}}
            kwargs["tools"] = tools_copy

        response = await self._client.messages.create(**kwargs)

        # Parse response
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        has_web_search = False

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )
            elif block.type in ("server_tool_use", "web_search_tool_result"):
                has_web_search = True

        content_text = "\n\n".join(text_parts) if text_parts else None

        # Capture raw content blocks when web search is present
        raw_content: list[dict[str, Any]] | None = None
        if has_web_search:
            raw_content = [
                block.model_dump() if hasattr(block, "model_dump") else {"type": block.type}
                for block in response.content
            ]

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_creation_input_tokens=(
                    getattr(response.usage, "cache_creation_input_tokens", 0) or 0
                ),
                cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            ),
            model=response.model,
            _raw_content=raw_content,
        )


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


class OpenAIProvider:
    """LLM provider using the OpenAI Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        default_model: str,
        *,
        _client: Any = None,
    ) -> None:
        self._default_model = default_model
        if _client is not None:
            self._client = _client
        else:
            import openai

            self._client = openai.AsyncOpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        translated = _messages_to_openai(messages)

        kwargs: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": translated,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        message = choice.message

        # Parse tool calls
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason,
            usage=Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
            model=response.model,
        )
