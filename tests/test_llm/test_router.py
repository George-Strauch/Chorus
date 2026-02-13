"""Tests for chorus.llm.router — interjection routing."""

from __future__ import annotations

from typing import Any

import pytest

from chorus.llm.providers import LLMResponse, Usage
from chorus.llm.router import (
    RouteDecision,
    create_router_provider,
    get_router_model,
    route_interjection,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=[],
        stop_reason="end_turn",
        usage=Usage(input_tokens=5, output_tokens=3),
        model="test",
    )


class FakeRouterProvider:
    """Fake provider that returns a scripted response."""

    def __init__(self, response: LLMResponse) -> None:
        self._response = response
        self.call_log: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        self.call_log.append({"messages": messages, "model": model})
        return self._response


class ErrorProvider:
    """Provider that always raises."""

    @property
    def provider_name(self) -> str:
        return "error"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        raise ConnectionError("API down")


# ---------------------------------------------------------------------------
# Route decision tests
# ---------------------------------------------------------------------------


class TestRouteInterjection:
    @pytest.mark.asyncio
    async def test_route_inject_on_followup(self) -> None:
        provider = FakeRouterProvider(_make_response("INJECT"))
        decision = await route_interjection(
            message="can you also fix the tests?",
            thread_summary="Working on auth refactor",
            current_step="Running bash: pytest",
            provider=provider,
            model="test",
        )
        assert decision is RouteDecision.INJECT

    @pytest.mark.asyncio
    async def test_route_new_thread_on_unrelated(self) -> None:
        provider = FakeRouterProvider(_make_response("NEW_THREAD"))
        decision = await route_interjection(
            message="what's the weather like?",
            thread_summary="Working on auth refactor",
            current_step="Running bash: pytest",
            provider=provider,
            model="test",
        )
        assert decision is RouteDecision.NEW_THREAD

    @pytest.mark.asyncio
    async def test_defaults_to_inject_on_error(self) -> None:
        decision = await route_interjection(
            message="fix the bug",
            thread_summary="Debugging",
            current_step="Calling LLM",
            provider=ErrorProvider(),
            model="test",
        )
        assert decision is RouteDecision.INJECT

    @pytest.mark.asyncio
    async def test_defaults_to_inject_on_ambiguous(self) -> None:
        provider = FakeRouterProvider(
            _make_response("I think this is a follow-up to the current conversation")
        )
        decision = await route_interjection(
            message="hmm okay",
            thread_summary="Debugging",
            current_step="Calling LLM",
            provider=provider,
            model="test",
        )
        assert decision is RouteDecision.INJECT

    @pytest.mark.asyncio
    async def test_defaults_to_inject_on_empty_response(self) -> None:
        provider = FakeRouterProvider(_make_response(""))
        decision = await route_interjection(
            message="test",
            thread_summary="Working",
            current_step="Step 1",
            provider=provider,
            model="test",
        )
        assert decision is RouteDecision.INJECT

    @pytest.mark.asyncio
    async def test_case_insensitive(self) -> None:
        provider = FakeRouterProvider(_make_response("new_thread"))
        decision = await route_interjection(
            message="unrelated question",
            thread_summary="Working",
            current_step="Step 1",
            provider=provider,
            model="test",
        )
        assert decision is RouteDecision.NEW_THREAD

    @pytest.mark.asyncio
    async def test_router_prompt_includes_context(self) -> None:
        provider = FakeRouterProvider(_make_response("INJECT"))
        await route_interjection(
            message="fix it",
            thread_summary="Auth refactor",
            current_step="Running pytest",
            provider=provider,
            model="test-model",
        )
        # Check the messages sent to the provider
        assert len(provider.call_log) == 1
        call = provider.call_log[0]
        assert call["model"] == "test-model"
        messages = call["messages"]
        # Should contain system prompt and user message with context
        all_content = " ".join(m.get("content", "") for m in messages)
        assert "Auth refactor" in all_content
        assert "Running pytest" in all_content
        assert "fix it" in all_content


# ---------------------------------------------------------------------------
# Router model selection
# ---------------------------------------------------------------------------


class TestGetRouterModel:
    def test_prefers_anthropic(self) -> None:
        provider_name, model = get_router_model(
            anthropic_key="sk-ant-test", openai_key="sk-openai-test"
        )
        assert provider_name == "anthropic"
        assert "haiku" in model

    def test_falls_back_to_openai(self) -> None:
        provider_name, model = get_router_model(
            anthropic_key=None, openai_key="sk-openai-test"
        )
        assert provider_name == "openai"
        assert "gpt-4o-mini" in model

    def test_raises_no_keys(self) -> None:
        with pytest.raises(ValueError, match="No API key"):
            get_router_model(anthropic_key=None, openai_key=None)


# ---------------------------------------------------------------------------
# Router provider creation
# ---------------------------------------------------------------------------


class TestCreateRouterProvider:
    def test_create_router_provider_anthropic(self) -> None:
        provider, model = create_router_provider(
            anthropic_key="sk-ant-test", openai_key=None
        )
        assert provider.provider_name == "anthropic"
        assert "haiku" in model

    def test_create_router_provider_openai(self) -> None:
        provider, model = create_router_provider(
            anthropic_key=None, openai_key="sk-openai-test"
        )
        assert provider.provider_name == "openai"
        assert "gpt-4o-mini" in model
