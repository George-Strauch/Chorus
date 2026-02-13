"""Tests for chorus.db_shell — CLI for querying the Chorus SQLite database."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

from chorus.db_shell import main
from chorus.storage.db import SCHEMA_SQL

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create a test database with schema and sample data."""
    db_file = tmp_path / "db" / "chorus.db"
    db_file.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db_file))
    conn.executescript(SCHEMA_SQL)
    # Insert sample agents
    conn.execute(
        "INSERT INTO agents (name, channel_id, guild_id, permissions, created_at, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("eve", 111, 999, "standard", "2026-01-01T00:00:00", "active"),
    )
    conn.execute(
        "INSERT INTO agents (name, channel_id, guild_id, permissions, created_at, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("alice", 222, 999, "open", "2026-01-02T00:00:00", "active"),
    )
    # Insert sample messages
    for i in range(25):
        conn.execute(
            "INSERT INTO messages (agent_name, role, content, timestamp) VALUES (?, ?, ?, ?)",
            ("eve", "user" if i % 2 == 0 else "assistant", f"msg {i}", f"2026-01-01T{i:02d}:00:00"),
        )
    conn.execute(
        "INSERT INTO messages (agent_name, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("eve", "user", "something went wrong with an error here", "2026-01-02T00:00:00"),
    )
    # Insert a session
    conn.execute(
        "INSERT INTO sessions (id, agent_name, description, saved_at, message_count, file_path) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("sess-1", "eve", "test session", "2026-01-01T00:00:00", 10, "/tmp/sess.json"),
    )
    conn.commit()
    conn.close()
    return db_file


class TestDbShellAgents:
    def test_list_agents(self, db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--db", str(db_path), "agents"])
        output = capsys.readouterr().out
        assert "eve" in output
        assert "alice" in output
        assert "2 agent(s)" in output

    def test_list_agents_empty(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        db_file = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_file))
        conn.executescript(SCHEMA_SQL)
        conn.close()
        main(["--db", str(db_file), "agents"])
        output = capsys.readouterr().out
        assert "No agents found" in output


class TestDbShellMessages:
    def test_messages_default(self, db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--db", str(db_path), "messages", "eve"])
        output = capsys.readouterr().out
        assert "20 message(s)" in output

    def test_messages_last_5(self, db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--db", str(db_path), "messages", "eve", "--last", "5"])
        output = capsys.readouterr().out
        assert "5 message(s)" in output

    def test_messages_search(self, db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--db", str(db_path), "messages", "eve", "--search", "error"])
        output = capsys.readouterr().out
        assert "error" in output
        assert "1 message(s)" in output

    def test_messages_unknown_agent(self, db_path: Path) -> None:
        with pytest.raises(SystemExit, match="1"):
            main(["--db", str(db_path), "messages", "nonexistent"])


class TestDbShellStatus:
    def test_status(self, db_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--db", str(db_path), "status"])
        output = capsys.readouterr().out
        assert "Agents:" in output
        assert "2" in output
        assert "Messages:" in output
        assert "26" in output
        assert "Sessions:" in output
        assert "1" in output


class TestDbShellNoCommand:
    def test_no_command_exits(self) -> None:
        with pytest.raises(SystemExit, match="1"):
            main([])

    def test_missing_db(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="1"):
            main(["--db", str(tmp_path / "nonexistent.db"), "agents"])
