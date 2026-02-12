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


class TestSessionStorage:
    def test_save_session(self) -> None:
        pytest.skip("Not implemented yet — TODO 007")

    def test_list_sessions(self) -> None:
        pytest.skip("Not implemented yet — TODO 007")

    def test_get_session(self) -> None:
        pytest.skip("Not implemented yet — TODO 007")


class TestAuditLog:
    def test_logs_action(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")

    def test_query_by_agent(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")
