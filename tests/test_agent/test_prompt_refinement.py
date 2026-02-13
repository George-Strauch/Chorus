"""Tests for chorus.agent.prompt_refinement — system prompt refinement on init."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from chorus.agent.prompt_refinement import (
    _create_refinement_provider,
    _pick_refinement_model,
    refine_system_prompt,
)

if TYPE_CHECKING:
    import pytest
from chorus.config import BotConfig
from chorus.llm.providers import LLMResponse, Usage

TEMPLATE_PROMPT = (
    "You are a general-purpose AI agent. You have access to a workspace directory "
    "where you can create, edit, and view files, run commands, and manage a git "
    "repository. Use your tools to accomplish tasks. Maintain notes about your "
    "workspace in your docs/ directory."
)


def _make_config(
    anthropic_key: str | None = None,
    openai_key: str | None = None,
) -> BotConfig:
    return BotConfig(
        discord_token="test-token",
        anthropic_api_key=anthropic_key,
        openai_api_key=openai_key,
    )


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=Usage(input_tokens=50, output_tokens=100),
        model="claude-haiku-4-5-20251001",
    )


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


class TestPickRefinementModel:
    def test_refine_picks_cheapest_available_model(self) -> None:
        """With both keys, picks Haiku (cheapest)."""
        result = _pick_refinement_model("sk-ant-test", "sk-openai-test")
        assert result is not None
        provider_name, model = result
        assert provider_name == "anthropic"
        assert "haiku" in model.lower()

    def test_refine_prefers_haiku_over_sonnet(self) -> None:
        """Anthropic key → Haiku model (not Sonnet)."""
        result = _pick_refinement_model("sk-ant-test", None)
        assert result is not None
        _, model = result
        assert "haiku" in model.lower()
        assert "sonnet" not in model.lower()

    def test_refine_falls_back_to_openai_mini_if_no_anthropic_key(self) -> None:
        """Only OpenAI key → gpt-4o-mini."""
        result = _pick_refinement_model(None, "sk-openai-test")
        assert result is not None
        provider_name, model = result
        assert provider_name == "openai"
        assert model == "gpt-4o-mini"

    def test_returns_none_when_no_keys(self) -> None:
        result = _pick_refinement_model(None, None)
        assert result is None


class TestCreateRefinementProvider:
    def test_creates_anthropic_provider(self) -> None:
        result = _create_refinement_provider("sk-ant-test", None)
        assert result is not None
        provider, model = result
        assert provider.provider_name == "anthropic"
        assert "haiku" in model.lower()

    def test_creates_openai_provider(self) -> None:
        result = _create_refinement_provider(None, "sk-openai-test")
        assert result is not None
        provider, model = result
        assert provider.provider_name == "openai"
        assert model == "gpt-4o-mini"

    def test_returns_none_when_no_keys(self) -> None:
        result = _create_refinement_provider(None, None)
        assert result is None


# ---------------------------------------------------------------------------
# Core refinement
# ---------------------------------------------------------------------------


class TestRefineSystemPrompt:
    async def test_refine_prompt_returns_refined_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mock provider returns refined text, verify it's returned."""
        refined_text = "You are a frontend specialist agent focused on React development."
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=_make_llm_response(refined_text))

        monkeypatch.setattr(
            "chorus.agent.prompt_refinement._create_refinement_provider",
            lambda *_args, **_kw: (mock_provider, "claude-haiku-4-5-20251001"),
        )

        config = _make_config(anthropic_key="sk-ant-test")
        result = await refine_system_prompt("frontend-bot", None, TEMPLATE_PROMPT, config)
        assert result == refined_text

    async def test_refine_prompt_includes_agent_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify agent name appears in the user message sent to the LLM."""
        captured_messages: list[Any] = []

        async def _capture_chat(messages: Any, **kwargs: Any) -> LLMResponse:
            captured_messages.extend(messages)
            return _make_llm_response("refined")

        mock_provider = MagicMock()
        mock_provider.chat = _capture_chat

        monkeypatch.setattr(
            "chorus.agent.prompt_refinement._create_refinement_provider",
            lambda *_args, **_kw: (mock_provider, "test-model"),
        )

        config = _make_config(anthropic_key="sk-ant-test")
        await refine_system_prompt("data-analyst-bot", None, TEMPLATE_PROMPT, config)

        user_msg = next(m for m in captured_messages if m["role"] == "user")
        assert "data-analyst-bot" in user_msg["content"]

    async def test_refine_prompt_preserves_template_structure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify template prompt is included in the LLM call."""
        captured_messages: list[Any] = []

        async def _capture_chat(messages: Any, **kwargs: Any) -> LLMResponse:
            captured_messages.extend(messages)
            return _make_llm_response("refined")

        mock_provider = MagicMock()
        mock_provider.chat = _capture_chat

        monkeypatch.setattr(
            "chorus.agent.prompt_refinement._create_refinement_provider",
            lambda *_args, **_kw: (mock_provider, "test-model"),
        )

        config = _make_config(anthropic_key="sk-ant-test")
        await refine_system_prompt("test-bot", None, TEMPLATE_PROMPT, config)

        user_msg = next(m for m in captured_messages if m["role"] == "user")
        assert TEMPLATE_PROMPT in user_msg["content"]

    async def test_refine_prompt_incorporates_user_description(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify description appears in user message."""
        captured_messages: list[Any] = []

        async def _capture_chat(messages: Any, **kwargs: Any) -> LLMResponse:
            captured_messages.extend(messages)
            return _make_llm_response("refined")

        mock_provider = MagicMock()
        mock_provider.chat = _capture_chat

        monkeypatch.setattr(
            "chorus.agent.prompt_refinement._create_refinement_provider",
            lambda *_args, **_kw: (mock_provider, "test-model"),
        )

        config = _make_config(anthropic_key="sk-ant-test")
        await refine_system_prompt(
            "docs-bot", "Manages project documentation and READMEs", TEMPLATE_PROMPT, config
        )

        user_msg = next(m for m in captured_messages if m["role"] == "user")
        assert "Manages project documentation and READMEs" in user_msg["content"]


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------


class TestRefineSystemPromptFallbacks:
    async def test_refine_prompt_returns_default_on_no_api_key(self) -> None:
        """Config with no keys → returns template unchanged."""
        config = _make_config()  # no keys
        result = await refine_system_prompt("test-bot", None, TEMPLATE_PROMPT, config)
        assert result == TEMPLATE_PROMPT

    async def test_refine_prompt_returns_default_on_llm_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """provider.chat raises → returns template unchanged."""
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(side_effect=RuntimeError("LLM API error"))

        monkeypatch.setattr(
            "chorus.agent.prompt_refinement._create_refinement_provider",
            lambda *_args, **_kw: (mock_provider, "test-model"),
        )

        config = _make_config(anthropic_key="sk-ant-test")
        result = await refine_system_prompt("test-bot", None, TEMPLATE_PROMPT, config)
        assert result == TEMPLATE_PROMPT

    async def test_refine_prompt_returns_default_on_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Slow provider → returns template unchanged."""

        async def _slow_chat(*args: Any, **kwargs: Any) -> LLMResponse:
            await asyncio.sleep(30)  # way over the 10s timeout
            return _make_llm_response("too late")

        mock_provider = MagicMock()
        mock_provider.chat = _slow_chat

        monkeypatch.setattr(
            "chorus.agent.prompt_refinement._create_refinement_provider",
            lambda *_args, **_kw: (mock_provider, "test-model"),
        )

        config = _make_config(anthropic_key="sk-ant-test")

        # Monkey-patch asyncio.wait_for to use a short timeout
        original_wait_for = asyncio.wait_for

        async def _fast_wait_for(coro: Any, timeout: float) -> Any:
            return await original_wait_for(coro, timeout=0.1)

        monkeypatch.setattr("chorus.agent.prompt_refinement.asyncio.wait_for", _fast_wait_for)

        result = await refine_system_prompt("test-bot", None, TEMPLATE_PROMPT, config)
        assert result == TEMPLATE_PROMPT

    async def test_refine_prompt_returns_default_on_empty_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty LLM response → returns template unchanged."""
        mock_provider = MagicMock()
        mock_provider.chat = AsyncMock(return_value=_make_llm_response(""))

        monkeypatch.setattr(
            "chorus.agent.prompt_refinement._create_refinement_provider",
            lambda *_args, **_kw: (mock_provider, "test-model"),
        )

        config = _make_config(anthropic_key="sk-ant-test")
        result = await refine_system_prompt("test-bot", None, TEMPLATE_PROMPT, config)
        assert result == TEMPLATE_PROMPT
