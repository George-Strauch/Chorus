"""Tests for chorus.storage.db — SQLite storage layer."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from chorus.storage.db import Database


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    db_path = tmp_path / "db" / "chorus.db"
    database = Database(db_path)
    await database.init()
    return database


class TestDatabaseInit:
    async def test_creates_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "db" / "chorus.db"
        db = Database(db_path)
        await db.init()
        try:
            async with db.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ) as cursor:
                tables = [row[0] for row in await cursor.fetchall()]
            assert "agents" in tables
            assert "sessions" in tables
            assert "audit_log" in tables
            assert "settings" in tables
            assert "messages" in tables
        finally:
            await db.close()

    async def test_idempotent_init(self, tmp_path: Path) -> None:
        db_path = tmp_path / "db" / "chorus.db"
        db = Database(db_path)
        await db.init()
        await db.init()  # Second call should not raise
        try:
            async with db.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ) as cursor:
                tables = [row[0] for row in await cursor.fetchall()]
            assert "agents" in tables
        finally:
            await db.close()


class TestAgentStorage:
    async def test_register_agent(self, db: Database) -> None:
        now = datetime.now(UTC).isoformat()
        await db.register_agent(
            name="test-bot",
            channel_id=12345,
            guild_id=99999,
            model="gpt-4o",
            permissions="standard",
            created_at=now,
        )
        agent = await db.get_agent("test-bot")
        assert agent is not None
        assert agent["name"] == "test-bot"
        assert agent["channel_id"] == 12345
        assert agent["guild_id"] == 99999
        assert agent["model"] == "gpt-4o"

    async def test_remove_agent(self, db: Database) -> None:
        now = datetime.now(UTC).isoformat()
        await db.register_agent(
            name="test-bot",
            channel_id=12345,
            guild_id=99999,
            model=None,
            permissions="standard",
            created_at=now,
        )
        await db.remove_agent("test-bot")
        agent = await db.get_agent("test-bot")
        assert agent is None

    async def test_list_agents(self, db: Database) -> None:
        now = datetime.now(UTC).isoformat()
        await db.register_agent("alpha", 100, 99999, None, "standard", now)
        await db.register_agent("beta", 200, 99999, None, "standard", now)
        agents = await db.list_agents(guild_id=99999)
        names = sorted(a["name"] for a in agents)
        assert names == ["alpha", "beta"]

    async def test_list_agents_filters_by_guild(self, db: Database) -> None:
        now = datetime.now(UTC).isoformat()
        await db.register_agent("agent-a", 100, 11111, None, "standard", now)
        await db.register_agent("agent-b", 200, 22222, None, "standard", now)
        agents = await db.list_agents(guild_id=11111)
        assert len(agents) == 1
        assert agents[0]["name"] == "agent-a"

    async def test_list_agents_no_guild_returns_all(self, db: Database) -> None:
        now = datetime.now(UTC).isoformat()
        await db.register_agent("agent-a", 100, 11111, None, "standard", now)
        await db.register_agent("agent-b", 200, 22222, None, "standard", now)
        agents = await db.list_agents()
        assert len(agents) == 2


class TestThreadStepStorage:
    async def test_persist_thread_step(self, db: Database) -> None:
        await db.persist_thread_step(
            agent_name="test-bot",
            thread_id=1,
            step_number=1,
            description="Calling LLM",
            started_at="2026-02-12T10:00:00",
            ended_at="2026-02-12T10:00:02",
            duration_ms=2000,
        )
        steps = await db.get_thread_steps("test-bot", 1)
        assert len(steps) == 1
        assert steps[0]["description"] == "Calling LLM"
        assert steps[0]["duration_ms"] == 2000

    async def test_get_thread_steps_ordered(self, db: Database) -> None:
        for i in range(1, 4):
            await db.persist_thread_step(
                agent_name="test-bot",
                thread_id=1,
                step_number=i,
                description=f"Step {i}",
                started_at=f"2026-02-12T10:00:0{i}",
            )
        steps = await db.get_thread_steps("test-bot", 1)
        assert len(steps) == 3
        assert [s["step_number"] for s in steps] == [1, 2, 3]

    async def test_get_agent_by_channel(self, db: Database) -> None:
        now = datetime.now(UTC).isoformat()
        await db.register_agent("test-bot", 12345, 99999, None, "standard", now)
        agent = await db.get_agent_by_channel(12345)
        assert agent is not None
        assert agent["name"] == "test-bot"

        missing = await db.get_agent_by_channel(99999)
        assert missing is None


class TestMessageStorage:
    async def test_persist_message(self, db: Database) -> None:
        await db.persist_message(
            agent_name="test-bot",
            role="user",
            timestamp="2026-02-12T10:00:00",
            content="Hello agent",
            thread_id=1,
        )
        msgs = await db.get_all_messages("test-bot")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello agent"
        assert msgs[0]["thread_id"] == 1

    async def test_persist_message_with_tool_calls(self, db: Database) -> None:
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "view"}}]
        await db.persist_message(
            agent_name="test-bot",
            role="assistant",
            timestamp="2026-02-12T10:00:01",
            content=None,
            tool_calls=tool_calls,
        )
        msgs = await db.get_all_messages("test-bot")
        assert len(msgs) == 1
        assert msgs[0]["tool_calls"] == tool_calls
        assert msgs[0]["content"] is None

    async def test_get_messages_since(self, db: Database) -> None:
        for i in range(5):
            await db.persist_message(
                agent_name="test-bot",
                role="user",
                timestamp=f"2026-02-12T10:00:0{i}",
                content=f"Message {i}",
            )
        msgs = await db.get_messages_since("test-bot", "2026-02-12T10:00:02")
        assert len(msgs) == 2
        assert msgs[0]["content"] == "Message 3"
        assert msgs[1]["content"] == "Message 4"

    async def test_get_messages_since_with_thread_id(self, db: Database) -> None:
        await db.persist_message(
            "test-bot", "user", "2026-02-12T10:00:00",
            content="Thread 1 msg", thread_id=1,
        )
        await db.persist_message(
            "test-bot", "user", "2026-02-12T10:00:01",
            content="Thread 2 msg", thread_id=2,
        )
        msgs = await db.get_messages_since(
            "test-bot", "2026-02-12T09:00:00", thread_id=1,
        )
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Thread 1 msg"

    async def test_get_all_messages_ordered(self, db: Database) -> None:
        await db.persist_message("test-bot", "user", "2026-02-12T10:00:02", content="Second")
        await db.persist_message("test-bot", "user", "2026-02-12T10:00:00", content="First")
        msgs = await db.get_all_messages("test-bot")
        assert msgs[0]["content"] == "First"
        assert msgs[1]["content"] == "Second"

    async def test_messages_isolated_by_agent(self, db: Database) -> None:
        await db.persist_message("agent-a", "user", "2026-02-12T10:00:00", content="A msg")
        await db.persist_message("agent-b", "user", "2026-02-12T10:00:01", content="B msg")
        msgs_a = await db.get_all_messages("agent-a")
        assert len(msgs_a) == 1
        assert msgs_a[0]["content"] == "A msg"


class TestClearTimeStorage:
    async def test_get_last_clear_time_default_none(self, db: Database) -> None:
        now = datetime.now(UTC).isoformat()
        await db.register_agent("test-bot", 12345, 99999, None, "standard", now)
        clear_time = await db.get_last_clear_time("test-bot")
        assert clear_time is None

    async def test_set_and_get_last_clear_time(self, db: Database) -> None:
        now = datetime.now(UTC).isoformat()
        await db.register_agent("test-bot", 12345, 99999, None, "standard", now)
        await db.set_last_clear_time("test-bot", "2026-02-12T12:00:00")
        clear_time = await db.get_last_clear_time("test-bot")
        assert clear_time == "2026-02-12T12:00:00"

    async def test_set_clear_time_updates(self, db: Database) -> None:
        now = datetime.now(UTC).isoformat()
        await db.register_agent("test-bot", 12345, 99999, None, "standard", now)
        await db.set_last_clear_time("test-bot", "2026-02-12T10:00:00")
        await db.set_last_clear_time("test-bot", "2026-02-12T14:00:00")
        clear_time = await db.get_last_clear_time("test-bot")
        assert clear_time == "2026-02-12T14:00:00"


class TestSessionStorage:
    async def test_save_session(self, db: Database) -> None:
        await db.save_session(
            session_id="sess-001",
            agent_name="test-bot",
            description="Auth refactor",
            summary="Worked on auth flow",
            saved_at="2026-02-12T14:00:00",
            message_count=42,
            file_path="/tmp/sessions/sess-001.json",
            window_start="2026-02-12T00:00:00",
            window_end="2026-02-12T14:00:00",
        )
        session = await db.get_session("sess-001")
        assert session is not None
        assert session["agent_name"] == "test-bot"
        assert session["description"] == "Auth refactor"
        assert session["summary"] == "Worked on auth flow"
        assert session["message_count"] == 42

    async def test_list_sessions(self, db: Database) -> None:
        for i in range(3):
            await db.save_session(
                session_id=f"sess-{i:03d}",
                agent_name="test-bot",
                description=f"Session {i}",
                summary=f"Summary {i}",
                saved_at=f"2026-02-12T1{i}:00:00",
                message_count=10 + i,
                file_path=f"/tmp/sessions/sess-{i:03d}.json",
                window_start=f"2026-02-12T0{i}:00:00",
                window_end=f"2026-02-12T1{i}:00:00",
            )
        sessions = await db.list_sessions("test-bot")
        assert len(sessions) == 3
        # Most recent first
        assert sessions[0]["session_id"] == "sess-002"
        assert sessions[2]["session_id"] == "sess-000"

    async def test_get_session_missing(self, db: Database) -> None:
        session = await db.get_session("nonexistent")
        assert session is None

    async def test_list_sessions_with_limit(self, db: Database) -> None:
        for i in range(5):
            await db.save_session(
                session_id=f"sess-{i:03d}",
                agent_name="test-bot",
                description=f"Session {i}",
                summary="",
                saved_at=f"2026-02-12T1{i}:00:00",
                message_count=10,
                file_path=f"/tmp/sess-{i:03d}.json",
                window_start="",
                window_end="",
            )
        sessions = await db.list_sessions("test-bot", limit=2)
        assert len(sessions) == 2


class TestAuditLog:
    def test_logs_action(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")

    def test_query_by_agent(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")
