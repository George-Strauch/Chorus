"""Tests for chorus.commands.agent_commands — /agent slash commands."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from chorus.commands.agent_commands import AgentCog


class TestPing:
    async def test_ping_command_returns_latency_message(self, mock_bot: MagicMock) -> None:
        mock_bot.latency = 0.045
        cog = AgentCog(mock_bot)

        interaction = MagicMock()
        interaction.response = AsyncMock()

        await cog.ping.callback(cog, interaction)

        interaction.response.send_message.assert_called_once()
        msg = interaction.response.send_message.call_args[0][0]
        assert "45" in msg
        assert "ms" in msg.lower()


class TestAgentInit:
    def test_creates_agent_and_channel(self) -> None:
        pytest.skip("Not implemented yet — TODO 001, 002")

    def test_uses_default_model(self) -> None:
        pytest.skip("Not implemented yet — TODO 009")

    def test_custom_system_prompt(self) -> None:
        pytest.skip("Not implemented yet — TODO 002")


class TestAgentDestroy:
    def test_removes_agent(self) -> None:
        pytest.skip("Not implemented yet — TODO 002")

    def test_keep_files_flag(self) -> None:
        pytest.skip("Not implemented yet — TODO 002")


class TestAgentList:
    def test_displays_agents(self) -> None:
        pytest.skip("Not implemented yet — TODO 002")


class TestAgentConfig:
    def test_updates_config(self) -> None:
        pytest.skip("Not implemented yet — TODO 002")
