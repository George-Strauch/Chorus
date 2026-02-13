"""Tests for chorus.commands.context_commands — /context slash commands."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

if TYPE_CHECKING:
    from pathlib import Path

    from chorus.storage.db import Database

from chorus.agent.context import ContextManager
from chorus.commands.context_commands import ContextCog


def _make_interaction(bot: MagicMock, channel_id: int = 987654321) -> MagicMock:
    interaction = MagicMock()
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.channel = MagicMock()
    interaction.channel.id = channel_id
    interaction.guild_id = 123456789
    return interaction


class TestContextClear:
    async def test_clear_advances_clear_time(
        self, mock_bot: MagicMock, context_db: Database, tmp_path: Path
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await context_db.register_agent("test-agent", 987654321, 99999, None, "standard", now)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        cm = ContextManager("test-agent", context_db, sessions_dir)
        await cm.persist_message(role="user", content="Before clear")

        mock_bot._context_managers = {"test-agent": cm}
        mock_bot._channel_to_agent = {987654321: "test-agent"}
        mock_bot.db = context_db

        cog = ContextCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.context_clear.callback(cog, interaction)

        interaction.response.send_message.assert_called_once()
        msgs = await cm.get_context()
        assert len(msgs) == 0

    async def test_clear_no_agent(self, mock_bot: MagicMock) -> None:
        mock_bot._context_managers = {}
        mock_bot._channel_to_agent = {}

        cog = ContextCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.context_clear.callback(cog, interaction)

        interaction.response.send_message.assert_called_once()
        msg = interaction.response.send_message.call_args[0][0]
        assert "no agent" in msg.lower()


class TestContextSave:
    async def test_save_creates_snapshot(
        self, mock_bot: MagicMock, context_db: Database, tmp_path: Path
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await context_db.register_agent("test-agent", 987654321, 99999, None, "standard", now)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        cm = ContextManager("test-agent", context_db, sessions_dir)
        await cm.persist_message(role="user", content="Save this")

        mock_bot._context_managers = {"test-agent": cm}
        mock_bot._channel_to_agent = {987654321: "test-agent"}
        mock_bot.db = context_db

        cog = ContextCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.context_save.callback(cog, interaction, description="Test snapshot")

        interaction.response.send_message.assert_called_once()
        sessions = await context_db.list_sessions("test-agent")
        assert len(sessions) == 1
        assert sessions[0]["description"] == "Test snapshot"


class TestContextHistory:
    async def test_history_returns_embed(
        self, mock_bot: MagicMock, context_db: Database, tmp_path: Path
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await context_db.register_agent("test-agent", 987654321, 99999, None, "standard", now)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        cm = ContextManager("test-agent", context_db, sessions_dir)
        await cm.persist_message(role="user", content="Work")
        await cm.save_snapshot(description="First session")

        mock_bot._context_managers = {"test-agent": cm}
        mock_bot._channel_to_agent = {987654321: "test-agent"}
        mock_bot.db = context_db

        cog = ContextCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.context_history.callback(cog, interaction)

        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args
        embed = call_kwargs.kwargs.get("embed")
        assert embed is not None


class TestContextRestore:
    async def test_restore_session(
        self, mock_bot: MagicMock, context_db: Database, tmp_path: Path
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await context_db.register_agent("test-agent", 987654321, 99999, None, "standard", now)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        cm = ContextManager("test-agent", context_db, sessions_dir)
        await cm.persist_message(role="user", content="Restore me")
        meta = await cm.save_snapshot(description="To restore")
        await cm.clear()

        mock_bot._context_managers = {"test-agent": cm}
        mock_bot._channel_to_agent = {987654321: "test-agent"}
        mock_bot.db = context_db

        cog = ContextCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.context_restore.callback(cog, interaction, session_id=meta.session_id)

        interaction.response.send_message.assert_called_once()
        msgs = await cm.get_context()
        contents = [m["content"] for m in msgs]
        assert "Restore me" in contents

    async def test_restore_nonexistent_session(
        self, mock_bot: MagicMock, context_db: Database, tmp_path: Path
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await context_db.register_agent("test-agent", 987654321, 99999, None, "standard", now)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        cm = ContextManager("test-agent", context_db, sessions_dir)

        mock_bot._context_managers = {"test-agent": cm}
        mock_bot._channel_to_agent = {987654321: "test-agent"}
        mock_bot.db = context_db

        cog = ContextCog(mock_bot)
        interaction = _make_interaction(mock_bot)
        await cog.context_restore.callback(cog, interaction, session_id="nonexistent")

        interaction.response.send_message.assert_called_once()
        msg = interaction.response.send_message.call_args[0][0]
        assert "not found" in msg.lower()
