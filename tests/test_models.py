"""Tests for chorus.models — data models."""

from __future__ import annotations

import pytest

from chorus.models import (
    Agent,
    InvalidAgentNameError,
    validate_agent_name,
)


class TestPermissionProfile:
    def test_from_preset_name(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")

    def test_serialization_roundtrip(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")


class TestAgentModel:
    def test_agent_from_json_dict(self) -> None:
        data = {
            "name": "my-agent",
            "channel_id": 123456,
            "model": "claude-sonnet-4-5-20250929",
            "system_prompt": "You are helpful.",
            "permissions": "open",
            "created_at": "2026-02-12T00:00:00+00:00",
        }
        agent = Agent.from_dict(data)
        assert agent.name == "my-agent"
        assert agent.channel_id == 123456
        assert agent.model == "claude-sonnet-4-5-20250929"
        assert agent.system_prompt == "You are helpful."
        assert agent.permissions == "open"
        assert agent.created_at == "2026-02-12T00:00:00+00:00"

    def test_agent_to_json_dict(self) -> None:
        agent = Agent(
            name="my-agent",
            channel_id=123456,
            model="claude-sonnet-4-5-20250929",
            system_prompt="You are helpful.",
            permissions="open",
            created_at="2026-02-12T00:00:00+00:00",
        )
        data = agent.to_dict()
        roundtripped = Agent.from_dict(data)
        assert roundtripped.name == agent.name
        assert roundtripped.channel_id == agent.channel_id
        assert roundtripped.model == agent.model
        assert roundtripped.system_prompt == agent.system_prompt
        assert roundtripped.permissions == agent.permissions

    def test_agent_defaults(self) -> None:
        agent = Agent(name="test-agent", channel_id=999)
        assert agent.model is None
        assert agent.permissions == "standard"
        assert agent.system_prompt is not None
        assert agent.created_at is not None

    def test_agent_rejects_invalid_name_uppercase(self) -> None:
        with pytest.raises(InvalidAgentNameError):
            validate_agent_name("Bad-Name")

    def test_agent_rejects_invalid_name_special_chars(self) -> None:
        with pytest.raises(InvalidAgentNameError):
            validate_agent_name("my_agent!")

    def test_agent_rejects_name_too_short(self) -> None:
        with pytest.raises(InvalidAgentNameError):
            validate_agent_name("a")

    def test_agent_rejects_name_too_long(self) -> None:
        with pytest.raises(InvalidAgentNameError):
            validate_agent_name("a" * 33)

    def test_agent_accepts_valid_hyphenated_name(self) -> None:
        validate_agent_name("my-cool-agent")  # Should not raise


class TestSession:
    def test_session_metadata(self) -> None:
        pytest.skip("Not implemented yet — TODO 007")
