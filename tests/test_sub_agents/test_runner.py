"""Tests for the sub-agent runner."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chorus.llm.providers import LLMResponse, Usage
from chorus.sub_agents.runner import (
    SubAgentResult,
    _create_provider,
    pick_cheap_model,
    run_sub_agent,
)

# ---------------------------------------------------------------------------
# SubAgentResult tests
# ---------------------------------------------------------------------------


class TestSubAgentResult:
    def test_success_result(self) -> None:
        usage = Usage(input_tokens=10, output_tokens=20)
        result = SubAgentResult(
            success=True,
            output="hello",
            model_used="claude-haiku-4-5-20251001",
            usage=usage,
        )
        assert result.success
        assert result.output == "hello"
        assert result.model_used == "claude-haiku-4-5-20251001"
        assert result.error is None

    def test_error_result(self) -> None:
        usage = Usage(input_tokens=0, output_tokens=0)
        result = SubAgentResult(
            success=False,
            output="",
            model_used="",
            usage=usage,
            error="No API keys",
        )
        assert not result.success
        assert result.error == "No API keys"


# ---------------------------------------------------------------------------
# pick_cheap_model tests
# ---------------------------------------------------------------------------


class TestPickCheapModel:
    def test_prefers_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        result = pick_cheap_model()
        assert result is not None
        env_var, provider, model = result
        assert env_var == "ANTHROPIC_API_KEY"
        assert provider == "anthropic"
        assert model == "claude-haiku-4-5-20251001"

    def test_falls_back_to_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        result = pick_cheap_model()
        assert result is not None
        env_var, provider, model = result
        assert env_var == "OPENAI_API_KEY"
        assert provider == "openai"
        assert model == "gpt-4o-mini"

    def test_returns_none_when_no_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert pick_cheap_model() is None

    def test_ignores_empty_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "  ")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        assert pick_cheap_model() is None


# ---------------------------------------------------------------------------
# _create_provider tests
# ---------------------------------------------------------------------------


class TestCreateProvider:
    def test_creates_anthropic_provider(self) -> None:
        from chorus.llm.providers import AnthropicProvider

        provider = _create_provider("anthropic", "sk-test", "claude-haiku")
        assert isinstance(provider, AnthropicProvider)

    def test_creates_openai_provider(self) -> None:
        from chorus.llm.providers import OpenAIProvider

        provider = _create_provider("openai", "sk-test", "gpt-4o-mini")
        assert isinstance(provider, OpenAIProvider)

    def test_unknown_defaults_to_openai(self) -> None:
        from chorus.llm.providers import OpenAIProvider

        provider = _create_provider("other", "sk-test", "some-model")
        assert isinstance(provider, OpenAIProvider)


# ---------------------------------------------------------------------------
# run_sub_agent tests
# ---------------------------------------------------------------------------


class TestRunSubAgent:
    @pytest.mark.asyncio
    async def test_no_api_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = await run_sub_agent(
            system_prompt="You are helpful.",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert not result.success
        assert "No API keys" in (result.error or "")

    @pytest.mark.asyncio
    async def test_successful_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        mock_response = LLMResponse(
            content="Generated output",
            model="claude-haiku-4-5-20251001",
            usage=Usage(input_tokens=10, output_tokens=5),
            tool_calls=[],
            stop_reason="end_turn",
        )
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=mock_response)

        with patch(
            "chorus.sub_agents.runner._create_provider",
            return_value=mock_provider,
        ):
            result = await run_sub_agent(
                system_prompt="You are helpful.",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result.success
        assert result.output == "Generated output"
        assert result.model_used == "claude-haiku-4-5-20251001"
        assert result.usage.input_tokens == 10

    @pytest.mark.asyncio
    async def test_system_prompt_prepended(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        mock_response = LLMResponse(
            content="ok",
            model="claude-haiku-4-5-20251001",
            usage=Usage(input_tokens=5, output_tokens=2),
            tool_calls=[],
            stop_reason="end_turn",
        )
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=mock_response)

        with patch(
            "chorus.sub_agents.runner._create_provider",
            return_value=mock_provider,
        ):
            await run_sub_agent(
                system_prompt="Be concise.",
                messages=[{"role": "user", "content": "hello"}],
            )

        # Verify system message was prepended
        call_args = mock_provider.chat.call_args
        msgs = call_args.kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Be concise."
        assert msgs[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_empty_system_prompt_not_prepended(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        mock_response = LLMResponse(
            content="ok",
            model="claude-haiku-4-5-20251001",
            usage=Usage(input_tokens=5, output_tokens=2),
            tool_calls=[],
            stop_reason="end_turn",
        )
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=mock_response)

        with patch(
            "chorus.sub_agents.runner._create_provider",
            return_value=mock_provider,
        ):
            await run_sub_agent(
                system_prompt="",
                messages=[{"role": "user", "content": "hello"}],
            )

        call_args = mock_provider.chat.call_args
        msgs = call_args.kwargs["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_empty_content_returns_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        mock_response = LLMResponse(
            content="",
            model="claude-haiku-4-5-20251001",
            usage=Usage(input_tokens=5, output_tokens=0),
            tool_calls=[],
            stop_reason="end_turn",
        )
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=mock_response)

        with patch(
            "chorus.sub_agents.runner._create_provider",
            return_value=mock_provider,
        ):
            result = await run_sub_agent(
                system_prompt="Be helpful.",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert not result.success
        assert "empty content" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        async def slow_chat(**kwargs: object) -> LLMResponse:
            await asyncio.sleep(10)
            return LLMResponse(
                content="late",
                model="claude-haiku",
                usage=Usage(input_tokens=0, output_tokens=0),
                tool_calls=[],
                stop_reason="end_turn",
            )

        mock_provider = MagicMock()
        mock_provider.chat = slow_chat

        with patch(
            "chorus.sub_agents.runner._create_provider",
            return_value=mock_provider,
        ):
            result = await run_sub_agent(
                system_prompt="Hi",
                messages=[{"role": "user", "content": "hi"}],
                timeout=0.01,
            )

        assert not result.success
        assert "Timeout" in (result.error or "")

    @pytest.mark.asyncio
    async def test_exception_handling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(
            side_effect=RuntimeError("connection lost")
        )

        with patch(
            "chorus.sub_agents.runner._create_provider",
            return_value=mock_provider,
        ):
            result = await run_sub_agent(
                system_prompt="Hi",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert not result.success
        assert "RuntimeError" in (result.error or "")
        assert "connection lost" in (result.error or "")

    @pytest.mark.asyncio
    async def test_model_override_matching_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        mock_response = LLMResponse(
            content="ok",
            model="claude-sonnet-4-20250514",
            usage=Usage(input_tokens=5, output_tokens=2),
            tool_calls=[],
            stop_reason="end_turn",
        )
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=mock_response)

        with patch(
            "chorus.sub_agents.runner._create_provider",
            return_value=mock_provider,
        ) as mock_create:
            await run_sub_agent(
                system_prompt="Hi",
                messages=[{"role": "user", "content": "hi"}],
                model_override="claude-sonnet-4-20250514",
            )

        # Verify override model was used
        mock_create.assert_called_once_with(
            "anthropic", "sk-ant-test", "claude-sonnet-4-20250514"
        )

    @pytest.mark.asyncio
    async def test_model_override_not_matching_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        mock_response = LLMResponse(
            content="ok",
            model="claude-haiku-4-5-20251001",
            usage=Usage(input_tokens=5, output_tokens=2),
            tool_calls=[],
            stop_reason="end_turn",
        )
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=mock_response)

        with patch(
            "chorus.sub_agents.runner._create_provider",
            return_value=mock_provider,
        ) as mock_create:
            await run_sub_agent(
                system_prompt="Hi",
                messages=[{"role": "user", "content": "hi"}],
                model_override="gpt-4o",  # Wrong provider
            )

        # Should fall back to default model
        mock_create.assert_called_once_with(
            "anthropic", "sk-ant-test", "claude-haiku-4-5-20251001"
        )
