"""Tests for chorus.llm.providers — multi-provider LLM client."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from chorus.llm.providers import (
    AnthropicProvider,
    LLMResponse,
    OpenAIProvider,
    Usage,
    _messages_to_anthropic,
    tools_to_anthropic,
    tools_to_openai,
)
from chorus.tools.registry import ToolDefinition

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TOOL = ToolDefinition(
    name="create_file",
    description="Create a file in the workspace.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "content": {"type": "string", "description": "File content"},
        },
        "required": ["path", "content"],
    },
    handler=AsyncMock(),
)


def _make_anthropic_text_response(
    text: str = "Hello!",
    model: str = "claude-sonnet-4-20250514",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> MagicMock:
    """Build a mock Anthropic Messages response with text only."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    resp = MagicMock()
    resp.content = [block]
    resp.stop_reason = "end_turn"
    resp.usage = usage
    resp.model = model
    return resp


def _make_anthropic_tool_response(
    tool_id: str = "toolu_01",
    tool_name: str = "create_file",
    arguments: dict[str, Any] | None = None,
    model: str = "claude-sonnet-4-20250514",
) -> MagicMock:
    """Build a mock Anthropic Messages response with a tool use block."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "I'll create a file."

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = tool_id
    tool_block.name = tool_name
    tool_block.input = arguments or {"path": "test.txt", "content": "hello"}

    usage = MagicMock()
    usage.input_tokens = 20
    usage.output_tokens = 15

    resp = MagicMock()
    resp.content = [text_block, tool_block]
    resp.stop_reason = "tool_use"
    resp.usage = usage
    resp.model = model
    return resp


def _make_openai_text_response(
    text: str = "Hello!",
    model: str = "gpt-4o",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response with text only."""
    message = MagicMock()
    message.content = text
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    usage.prompt_tokens_details = None  # No caching by default

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model = model
    return resp


def _make_openai_tool_response(
    tool_id: str = "call_01",
    tool_name: str = "create_file",
    arguments: dict[str, Any] | None = None,
    model: str = "gpt-4o",
) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response with a function call."""
    func = MagicMock()
    func.name = tool_name
    func.arguments = json.dumps(arguments or {"path": "test.txt", "content": "hello"})

    tc = MagicMock()
    tc.id = tool_id
    tc.type = "function"
    tc.function = func

    message = MagicMock()
    message.content = "I'll create a file."
    message.tool_calls = [tc]

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "tool_calls"

    usage = MagicMock()
    usage.prompt_tokens = 20
    usage.completion_tokens = 15
    usage.prompt_tokens_details = None  # No caching by default

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model = model
    return resp


# ---------------------------------------------------------------------------
# Usage dataclass
# ---------------------------------------------------------------------------


class TestUsage:
    def test_creation(self) -> None:
        u = Usage(input_tokens=10, output_tokens=5)
        assert u.input_tokens == 10
        assert u.output_tokens == 5

    def test_creation_with_cache_fields(self) -> None:
        u = Usage(
            input_tokens=10, output_tokens=5,
            cache_creation_input_tokens=100, cache_read_input_tokens=200,
        )
        assert u.cache_creation_input_tokens == 100
        assert u.cache_read_input_tokens == 200

    def test_cache_fields_default_zero(self) -> None:
        u = Usage(input_tokens=10, output_tokens=5)
        assert u.cache_creation_input_tokens == 0
        assert u.cache_read_input_tokens == 0

    def test_add(self) -> None:
        u1 = Usage(input_tokens=10, output_tokens=5)
        u2 = Usage(input_tokens=20, output_tokens=10)
        total = u1 + u2
        assert total.input_tokens == 30
        assert total.output_tokens == 15

    def test_add_with_cache_fields(self) -> None:
        u1 = Usage(
            input_tokens=10, output_tokens=5,
            cache_creation_input_tokens=100, cache_read_input_tokens=200,
        )
        u2 = Usage(
            input_tokens=20, output_tokens=10,
            cache_creation_input_tokens=50, cache_read_input_tokens=300,
        )
        total = u1 + u2
        assert total.cache_creation_input_tokens == 150
        assert total.cache_read_input_tokens == 500


# ---------------------------------------------------------------------------
# Tool schema translation
# ---------------------------------------------------------------------------


class TestToolSchemaTranslation:
    def test_tools_to_anthropic(self) -> None:
        result = tools_to_anthropic([SAMPLE_TOOL])
        assert len(result) == 1
        tool = result[0]
        assert tool["name"] == "create_file"
        assert tool["description"] == "Create a file in the workspace."
        assert tool["input_schema"] == SAMPLE_TOOL.parameters

    def test_tools_to_openai(self) -> None:
        result = tools_to_openai([SAMPLE_TOOL])
        assert len(result) == 1
        tool = result[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "create_file"
        assert tool["function"]["description"] == "Create a file in the workspace."
        assert tool["function"]["parameters"] == SAMPLE_TOOL.parameters

    def test_parameter_types_preserved(self) -> None:
        """Both translations preserve the JSON Schema parameter types exactly."""
        anth = tools_to_anthropic([SAMPLE_TOOL])[0]
        oai = tools_to_openai([SAMPLE_TOOL])[0]
        assert anth["input_schema"]["properties"]["path"]["type"] == "string"
        assert oai["function"]["parameters"]["properties"]["path"]["type"] == "string"

    def test_empty_tools_list(self) -> None:
        assert tools_to_anthropic([]) == []
        assert tools_to_openai([]) == []


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    def test_provider_name(self) -> None:
        mock_client = MagicMock()
        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        assert provider.provider_name == "anthropic"

    @pytest.mark.asyncio
    async def test_parses_text_response(self) -> None:
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_make_anthropic_text_response())

        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.tool_calls == []
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_parses_tool_use_response(self) -> None:
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_make_anthropic_tool_response())

        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        result = await provider.chat(
            messages=[{"role": "user", "content": "Create a file"}],
        )

        assert result.stop_reason == "tool_use"
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "toolu_01"
        assert tc.name == "create_file"
        assert tc.arguments == {"path": "test.txt", "content": "hello"}
        assert result.content == "I'll create a file."

    @pytest.mark.asyncio
    async def test_sends_correct_request_format(self) -> None:
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_make_anthropic_text_response())

        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        tools_schema = tools_to_anthropic([SAMPLE_TOOL])
        await provider.chat(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
            tools=tools_schema,
            model="claude-sonnet-4-20250514",
        )

        call_kwargs = mock_client.messages.create.call_args
        # System prompt should be extracted and passed as block format with cache_control
        system = call_kwargs.kwargs["system"]
        assert isinstance(system, list)
        assert system[0]["text"] == "You are helpful."
        # Messages should NOT include the system message
        msgs = call_kwargs.kwargs["messages"]
        assert all(m["role"] != "system" for m in msgs)
        # Tools should include cache_control on last tool
        sent_tools = call_kwargs.kwargs["tools"]
        assert sent_tools[-1]["cache_control"] == {"type": "ephemeral"}
        # Original tool data should be preserved
        assert sent_tools[0]["name"] == tools_schema[0]["name"]
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_handles_system_prompt(self) -> None:
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_make_anthropic_text_response())

        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        await provider.chat(
            messages=[
                {"role": "system", "content": "System msg 1"},
                {"role": "system", "content": "System msg 2"},
                {"role": "user", "content": "Hi"},
            ],
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        # Multiple system messages should be joined into a single block
        system = call_kwargs["system"]
        assert isinstance(system, list)
        assert "System msg 1" in system[0]["text"]
        assert "System msg 2" in system[0]["text"]

    @pytest.mark.asyncio
    async def test_translates_tool_result_messages(self) -> None:
        """Tool results in neutral format are translated to Anthropic format."""
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_make_anthropic_text_response())

        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        await provider.chat(
            messages=[
                {"role": "user", "content": "Create a file"},
                {
                    "role": "assistant",
                    "content": "Creating file.",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "name": "create_file",
                            "arguments": {"path": "a.txt", "content": "hi"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "tc_1", "content": "File created"},
            ],
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        msgs = call_kwargs["messages"]

        # Assistant message should have tool_use blocks in content
        assistant_msg = msgs[1]
        assert assistant_msg["role"] == "assistant"
        assert any(b.get("type") == "tool_use" for b in assistant_msg["content"])

        # Tool result should become a user message with tool_result blocks
        tool_msg = msgs[2]
        assert tool_msg["role"] == "user"
        assert any(b.get("type") == "tool_result" for b in tool_msg["content"])

    @pytest.mark.asyncio
    async def test_rate_limit_error_propagates(self) -> None:
        mock_client = MagicMock()
        mock_client.messages = MagicMock()

        import anthropic

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.RateLimitError(
                message="Rate limited",
                response=mock_response,
                body=None,
            )
        )

        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        with pytest.raises(anthropic.RateLimitError):
            await provider.chat(messages=[{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_auth_error_propagates(self) -> None:
        mock_client = MagicMock()
        mock_client.messages = MagicMock()

        import anthropic

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.AuthenticationError(
                message="Invalid API key",
                response=mock_response,
                body=None,
            )
        )

        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        with pytest.raises(anthropic.AuthenticationError):
            await provider.chat(messages=[{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_uses_default_model(self) -> None:
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_make_anthropic_text_response())

        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        await provider.chat(messages=[{"role": "user", "content": "Hi"}])

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    def test_provider_name(self) -> None:
        mock_client = MagicMock()
        provider = OpenAIProvider("sk-test", "gpt-4o", _client=mock_client)
        assert provider.provider_name == "openai"

    @pytest.mark.asyncio
    async def test_parses_text_response(self) -> None:
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_openai_text_response()
        )

        provider = OpenAIProvider("sk-test", "gpt-4o", _client=mock_client)
        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.tool_calls == []
        assert result.stop_reason == "stop"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_parses_tool_call_response(self) -> None:
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_openai_tool_response()
        )

        provider = OpenAIProvider("sk-test", "gpt-4o", _client=mock_client)
        result = await provider.chat(
            messages=[{"role": "user", "content": "Create a file"}],
        )

        assert result.stop_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "call_01"
        assert tc.name == "create_file"
        assert tc.arguments == {"path": "test.txt", "content": "hello"}

    @pytest.mark.asyncio
    async def test_sends_correct_request_format(self) -> None:
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_openai_text_response()
        )

        provider = OpenAIProvider("sk-test", "gpt-4o", _client=mock_client)
        tools_schema = tools_to_openai([SAMPLE_TOOL])
        await provider.chat(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
            tools=tools_schema,
            model="gpt-4o",
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        # System messages pass through for OpenAI
        msgs = call_kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert call_kwargs["tools"] == tools_schema
        assert call_kwargs["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_includes_system_prompt_as_system_message(self) -> None:
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_openai_text_response()
        )

        provider = OpenAIProvider("sk-test", "gpt-4o", _client=mock_client)
        await provider.chat(
            messages=[
                {"role": "system", "content": "You are a helper."},
                {"role": "user", "content": "Hi"},
            ],
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        msgs = call_kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "You are a helper."}

    @pytest.mark.asyncio
    async def test_translates_tool_result_messages(self) -> None:
        """Tool results in neutral format pass through to OpenAI format."""
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_openai_text_response()
        )

        provider = OpenAIProvider("sk-test", "gpt-4o", _client=mock_client)
        await provider.chat(
            messages=[
                {"role": "user", "content": "Create a file"},
                {
                    "role": "assistant",
                    "content": "Creating file.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "create_file",
                            "arguments": {"path": "a.txt", "content": "hi"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "File created"},
            ],
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        msgs = call_kwargs["messages"]

        # Assistant message should have tool_calls in OpenAI format
        assistant_msg = msgs[1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["tool_calls"][0]["type"] == "function"
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "create_file"

        # Tool result keeps role "tool"
        tool_msg = msgs[2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_1"

    @pytest.mark.asyncio
    async def test_rate_limit_error_propagates(self) -> None:
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()

        import openai

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.json.return_value = {"error": {"message": "Rate limited"}}
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.RateLimitError(
                message="Rate limited",
                response=mock_response,
                body=None,
            )
        )

        provider = OpenAIProvider("sk-test", "gpt-4o", _client=mock_client)
        with pytest.raises(openai.RateLimitError):
            await provider.chat(messages=[{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_auth_error_propagates(self) -> None:
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()

        import openai

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_client.chat.completions.create = AsyncMock(
            side_effect=openai.AuthenticationError(
                message="Invalid API key",
                response=mock_response,
                body=None,
            )
        )

        provider = OpenAIProvider("sk-test", "gpt-4o", _client=mock_client)
        with pytest.raises(openai.AuthenticationError):
            await provider.chat(messages=[{"role": "user", "content": "Hi"}])

    @pytest.mark.asyncio
    async def test_uses_default_model(self) -> None:
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_openai_text_response()
        )

        provider = OpenAIProvider("sk-test", "gpt-4o", _client=mock_client)
        await provider.chat(messages=[{"role": "user", "content": "Hi"}])

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# Protocol compatibility
# ---------------------------------------------------------------------------


class TestProtocolCompatibility:
    def test_anthropic_provider_satisfies_protocol(self) -> None:
        mock_client = MagicMock()
        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        # Protocol has provider_name and chat
        assert hasattr(provider, "provider_name")
        assert hasattr(provider, "chat")
        assert provider.provider_name == "anthropic"

    def test_openai_provider_satisfies_protocol(self) -> None:
        mock_client = MagicMock()
        provider = OpenAIProvider("sk-test", "gpt-4o", _client=mock_client)
        assert hasattr(provider, "provider_name")
        assert hasattr(provider, "chat")
        assert provider.provider_name == "openai"

    @pytest.mark.asyncio
    async def test_both_providers_return_same_response_shape(self) -> None:
        """Both providers produce LLMResponse with identical field structure."""
        anth_client = MagicMock()
        anth_client.messages = MagicMock()
        anth_client.messages.create = AsyncMock(return_value=_make_anthropic_text_response())
        oai_client = MagicMock()
        oai_client.chat = MagicMock()
        oai_client.chat.completions = MagicMock()
        oai_client.chat.completions.create = AsyncMock(return_value=_make_openai_text_response())

        anth = AnthropicProvider("sk-ant", "claude-sonnet-4-20250514", _client=anth_client)
        oai = OpenAIProvider("sk-oai", "gpt-4o", _client=oai_client)

        r1 = await anth.chat(messages=[{"role": "user", "content": "Hi"}])
        r2 = await oai.chat(messages=[{"role": "user", "content": "Hi"}])

        for r in (r1, r2):
            assert isinstance(r, LLMResponse)
            assert isinstance(r.content, str)
            assert isinstance(r.tool_calls, list)
            assert isinstance(r.stop_reason, str)
            assert isinstance(r.usage, Usage)
            assert isinstance(r.model, str)


# ---------------------------------------------------------------------------
# Web search support
# ---------------------------------------------------------------------------


def _make_anthropic_web_search_response(
    model: str = "claude-sonnet-4-20250514",
) -> MagicMock:
    """Build a mock Anthropic response with web search blocks."""
    text_block1 = MagicMock()
    text_block1.type = "text"
    text_block1.text = "Let me search for that."
    text_block1.model_dump = MagicMock(
        return_value={"type": "text", "text": "Let me search for that."}
    )

    server_tool_block = MagicMock()
    server_tool_block.type = "server_tool_use"
    server_tool_block.id = "srvtoolu_01"
    server_tool_block.name = "web_search"
    server_tool_block.model_dump = MagicMock(
        return_value={
            "type": "server_tool_use",
            "id": "srvtoolu_01",
            "name": "web_search",
            "input": {"query": "latest news"},
        }
    )

    web_result_block = MagicMock()
    web_result_block.type = "web_search_tool_result"
    web_result_block.model_dump = MagicMock(
        return_value={
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_01",
            "content": [{"type": "web_search_result", "url": "https://example.com"}],
        }
    )

    text_block2 = MagicMock()
    text_block2.type = "text"
    text_block2.text = "Here are the results."
    text_block2.model_dump = MagicMock(
        return_value={"type": "text", "text": "Here are the results."}
    )

    usage = MagicMock()
    usage.input_tokens = 30
    usage.output_tokens = 20

    resp = MagicMock()
    resp.content = [text_block1, server_tool_block, web_result_block, text_block2]
    resp.stop_reason = "end_turn"
    resp.usage = usage
    resp.model = model
    return resp


class TestAnthropicWebSearch:
    @pytest.mark.asyncio
    async def test_concatenates_multiple_text_blocks(self) -> None:
        """Multiple text blocks are joined with double newline."""
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_anthropic_web_search_response()
        )

        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        result = await provider.chat(messages=[{"role": "user", "content": "Search for news"}])

        assert result.content is not None
        assert "Let me search for that." in result.content
        assert "Here are the results." in result.content
        assert "\n\n" in result.content

    @pytest.mark.asyncio
    async def test_raw_content_set_for_web_search_blocks(self) -> None:
        """_raw_content is populated when web search blocks are present."""
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_anthropic_web_search_response()
        )

        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        result = await provider.chat(messages=[{"role": "user", "content": "Search"}])

        assert result._raw_content is not None
        assert len(result._raw_content) == 4
        types = [b["type"] for b in result._raw_content]
        assert "server_tool_use" in types
        assert "web_search_tool_result" in types

    @pytest.mark.asyncio
    async def test_raw_content_none_without_web_search(self) -> None:
        """_raw_content is None for normal responses without web search."""
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_make_anthropic_text_response())

        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        result = await provider.chat(messages=[{"role": "user", "content": "Hi"}])

        assert result._raw_content is None


class TestAnthropicPromptCaching:
    @pytest.mark.asyncio
    async def test_cache_control_on_system_prompt(self) -> None:
        """System prompt is sent as block format with cache_control."""
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_make_anthropic_text_response())

        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        await provider.chat(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        system = call_kwargs["system"]
        assert isinstance(system, list)
        assert len(system) == 1
        assert system[0]["type"] == "text"
        assert system[0]["text"] == "You are helpful."
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_cache_control_on_tools(self) -> None:
        """Last tool gets cache_control; others don't. Original list not mutated."""
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_make_anthropic_text_response())

        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        tool_a = {"name": "tool_a", "description": "A", "input_schema": {}}
        tool_b = {"name": "tool_b", "description": "B", "input_schema": {}}
        original_tools = [tool_a, tool_b]

        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=original_tools,
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        sent_tools = call_kwargs["tools"]
        # Last tool should have cache_control
        assert sent_tools[-1]["cache_control"] == {"type": "ephemeral"}
        # First tool should NOT have cache_control
        assert "cache_control" not in sent_tools[0]
        # Original tools should NOT be mutated
        assert "cache_control" not in tool_a
        assert "cache_control" not in tool_b

    @pytest.mark.asyncio
    async def test_cache_control_single_tool(self) -> None:
        """Single tool gets cache_control."""
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=_make_anthropic_text_response())

        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[{"name": "only_tool", "description": "X", "input_schema": {}}],
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["tools"][0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_cache_usage_parsed(self) -> None:
        """Cache usage fields are extracted from Anthropic response."""
        mock_client = MagicMock()
        mock_client.messages = MagicMock()

        resp = _make_anthropic_text_response()
        resp.usage.cache_creation_input_tokens = 1500
        resp.usage.cache_read_input_tokens = 8000
        mock_client.messages.create = AsyncMock(return_value=resp)

        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        result = await provider.chat(messages=[{"role": "user", "content": "Hi"}])

        assert result.usage.cache_creation_input_tokens == 1500
        assert result.usage.cache_read_input_tokens == 8000

    @pytest.mark.asyncio
    async def test_cache_usage_defaults_zero(self) -> None:
        """Missing cache fields default to 0."""
        mock_client = MagicMock()
        mock_client.messages = MagicMock()

        resp = _make_anthropic_text_response()
        # Simulate old API response without cache fields
        del resp.usage.cache_creation_input_tokens
        del resp.usage.cache_read_input_tokens
        mock_client.messages.create = AsyncMock(return_value=resp)

        provider = AnthropicProvider("sk-ant-test", "claude-sonnet-4-20250514", _client=mock_client)
        result = await provider.chat(messages=[{"role": "user", "content": "Hi"}])

        assert result.usage.cache_creation_input_tokens == 0
        assert result.usage.cache_read_input_tokens == 0


class TestMessagesToAnthropicWebSearch:
    def test_uses_anthropic_content_when_present(self) -> None:
        """_anthropic_content on an assistant message is used directly."""
        raw_blocks = [
            {"type": "text", "text": "Searching..."},
            {"type": "server_tool_use", "id": "srv_1", "name": "web_search"},
            {"type": "web_search_tool_result", "tool_use_id": "srv_1", "content": []},
            {"type": "text", "text": "Found results."},
        ]
        messages = [
            {"role": "user", "content": "Search something"},
            {
                "role": "assistant",
                "content": "Searching...\n\nFound results.",
                "_anthropic_content": raw_blocks,
            },
        ]
        _, translated = _messages_to_anthropic(messages)
        # The assistant message should use the raw blocks directly
        assert translated[1]["role"] == "assistant"
        assert translated[1]["content"] == raw_blocks

    def test_fallback_without_anthropic_content(self) -> None:
        """Without _anthropic_content, normal pass-through behavior."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        _, translated = _messages_to_anthropic(messages)
        assert translated[1]["role"] == "assistant"
        assert translated[1]["content"] == "Hello!"
