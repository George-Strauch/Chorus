"""Tests for chorus.storage.db — SQLite storage layer."""

from pathlib import Path

import pytest

from chorus.storage.db import Database


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
    def test_register_agent(self) -> None:
        pytest.skip("Not implemented yet — TODO 002")

    def test_unregister_agent(self) -> None:
        pytest.skip("Not implemented yet — TODO 002")

    def test_list_agents(self) -> None:
        pytest.skip("Not implemented yet — TODO 002")


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
