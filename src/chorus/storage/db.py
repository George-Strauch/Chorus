"""Async SQLite storage layer for Chorus."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import aiosqlite

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("chorus.storage.db")

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS agents (
    name TEXT PRIMARY KEY,
    channel_id INTEGER UNIQUE NOT NULL,
    guild_id INTEGER NOT NULL,
    model TEXT,
    permissions TEXT NOT NULL DEFAULT 'standard',
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    last_clear_time TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL REFERENCES agents(name),
    description TEXT,
    summary TEXT,
    saved_at TEXT NOT NULL,
    message_count INTEGER,
    file_path TEXT NOT NULL,
    window_start TEXT,
    window_end TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    action_string TEXT NOT NULL,
    decision TEXT NOT NULL,
    user_id INTEGER,
    detail TEXT
);

CREATE TABLE IF NOT EXISTS settings (
    guild_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (guild_id, key)
);

CREATE TABLE IF NOT EXISTS thread_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    thread_id INTEGER NOT NULL,
    step_number INTEGER NOT NULL,
    description TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_thread_steps_agent_thread
    ON thread_steps(agent_name, thread_id);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    thread_id INTEGER,
    role TEXT NOT NULL,
    content TEXT,
    tool_calls TEXT,
    tool_call_id TEXT,
    timestamp TEXT NOT NULL,
    discord_message_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_messages_agent_time
    ON messages(agent_name, timestamp);
"""


class Database:
    """Async SQLite database wrapper for Chorus state."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None

    async def init(self) -> None:
        """Create parent directories, open connection, run schema and migrations."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path)
        await self._conn.executescript(SCHEMA_SQL)
        await self._conn.commit()
        await self._migrate()
        logger.info("Database initialized at %s", self._path)

    async def _migrate(self) -> None:
        """Run idempotent schema migrations for columns added after initial release."""
        conn = self.connection
        migrations: list[tuple[str, str, str]] = [
            ("agents", "last_clear_time", "ALTER TABLE agents ADD COLUMN last_clear_time TEXT"),
            ("sessions", "summary", "ALTER TABLE sessions ADD COLUMN summary TEXT"),
            ("sessions", "window_start", "ALTER TABLE sessions ADD COLUMN window_start TEXT"),
            ("sessions", "window_end", "ALTER TABLE sessions ADD COLUMN window_end TEXT"),
        ]
        for table, column, sql in migrations:
            try:
                await conn.execute(f"SELECT {column} FROM {table} LIMIT 0")  # noqa: S608
            except aiosqlite.OperationalError:
                await conn.execute(sql)
                logger.info("Migration: added %s.%s", table, column)
        await conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def connection(self) -> aiosqlite.Connection:
        """Return the active connection. Raises if not initialized."""
        if self._conn is None:
            raise RuntimeError("Database not initialized — call init() first")
        return self._conn

    async def register_agent(
        self,
        name: str,
        channel_id: int,
        guild_id: int,
        model: str | None,
        permissions: str,
        created_at: str,
    ) -> None:
        """Insert a new agent row."""
        await self.connection.execute(
            "INSERT INTO agents (name, channel_id, guild_id, model, permissions, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, channel_id, guild_id, model, permissions, created_at),
        )
        await self.connection.commit()
        logger.info("Registered agent %s (channel=%d, guild=%d)", name, channel_id, guild_id)

    async def remove_agent(self, name: str) -> None:
        """Delete an agent row by name."""
        await self.connection.execute("DELETE FROM agents WHERE name = ?", (name,))
        await self.connection.commit()
        logger.info("Removed agent %s from database", name)

    async def get_agent(self, name: str) -> dict[str, object] | None:
        """Fetch a single agent by name, or None if not found."""
        async with self.connection.execute(
            "SELECT name, channel_id, guild_id, model, permissions, created_at, status "
            "FROM agents WHERE name = ?",
            (name,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "name": row[0],
            "channel_id": row[1],
            "guild_id": row[2],
            "model": row[3],
            "permissions": row[4],
            "created_at": row[5],
            "status": row[6],
        }

    async def list_agents(self, guild_id: int | None = None) -> list[dict[str, object]]:
        """List all agents, optionally filtered by guild."""
        if guild_id is not None:
            query = (
                "SELECT name, channel_id, guild_id, model, permissions, created_at, status "
                "FROM agents WHERE guild_id = ?"
            )
            params: tuple[object, ...] = (guild_id,)
        else:
            query = (
                "SELECT name, channel_id, guild_id, model, permissions, created_at, status "
                "FROM agents"
            )
            params = ()
        async with self.connection.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                "name": row[0],
                "channel_id": row[1],
                "guild_id": row[2],
                "model": row[3],
                "permissions": row[4],
                "created_at": row[5],
                "status": row[6],
            }
            for row in rows
        ]

    async def get_agent_by_channel(self, channel_id: int) -> dict[str, object] | None:
        """Fetch a single agent by channel ID, or None if not found."""
        async with self.connection.execute(
            "SELECT name, channel_id, guild_id, model, permissions, created_at, status "
            "FROM agents WHERE channel_id = ?",
            (channel_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "name": row[0],
            "channel_id": row[1],
            "guild_id": row[2],
            "model": row[3],
            "permissions": row[4],
            "created_at": row[5],
            "status": row[6],
        }

    async def persist_thread_step(
        self,
        agent_name: str,
        thread_id: int,
        step_number: int,
        description: str,
        started_at: str,
        ended_at: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """Insert a thread step record."""
        await self.connection.execute(
            "INSERT INTO thread_steps "
            "(agent_name, thread_id, step_number, description, started_at, ended_at, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (agent_name, thread_id, step_number, description, started_at, ended_at, duration_ms),
        )
        await self.connection.commit()

    async def get_thread_steps(
        self, agent_name: str, thread_id: int
    ) -> list[dict[str, object]]:
        """Get all steps for a thread, ordered by step_number."""
        async with self.connection.execute(
            "SELECT step_number, description, started_at, ended_at, duration_ms "
            "FROM thread_steps WHERE agent_name = ? AND thread_id = ? "
            "ORDER BY step_number",
            (agent_name, thread_id),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                "step_number": row[0],
                "description": row[1],
                "started_at": row[2],
                "ended_at": row[3],
                "duration_ms": row[4],
            }
            for row in rows
        ]

    # ── Message persistence ──────────────────────────────────────────────

    async def persist_message(
        self,
        agent_name: str,
        role: str,
        timestamp: str,
        *,
        thread_id: int | None = None,
        content: str | None = None,
        tool_calls: list[Any] | None = None,
        tool_call_id: str | None = None,
        discord_message_id: int | None = None,
    ) -> None:
        """Insert a message row."""
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        await self.connection.execute(
            "INSERT INTO messages "
            "(agent_name, thread_id, role, content, tool_calls, tool_call_id, "
            "timestamp, discord_message_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                agent_name,
                thread_id,
                role,
                content,
                tool_calls_json,
                tool_call_id,
                timestamp,
                discord_message_id,
            ),
        )
        await self.connection.commit()

    async def get_messages_since(
        self,
        agent_name: str,
        since: str,
        *,
        thread_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return messages after a given ISO timestamp, ordered by time."""
        if thread_id is not None:
            query = (
                "SELECT id, agent_name, thread_id, role, content, tool_calls, "
                "tool_call_id, timestamp, discord_message_id "
                "FROM messages WHERE agent_name = ? AND timestamp > ? AND thread_id = ? "
                "ORDER BY timestamp ASC"
            )
            params: tuple[Any, ...] = (agent_name, since, thread_id)
        else:
            query = (
                "SELECT id, agent_name, thread_id, role, content, tool_calls, "
                "tool_call_id, timestamp, discord_message_id "
                "FROM messages WHERE agent_name = ? AND timestamp > ? "
                "ORDER BY timestamp ASC"
            )
            params = (agent_name, since)
        async with self.connection.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_message(row) for row in rows]

    async def get_all_messages(self, agent_name: str) -> list[dict[str, Any]]:
        """Return all messages for an agent, ordered by time."""
        async with self.connection.execute(
            "SELECT id, agent_name, thread_id, role, content, tool_calls, "
            "tool_call_id, timestamp, discord_message_id "
            "FROM messages WHERE agent_name = ? ORDER BY timestamp ASC",
            (agent_name,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_message(row) for row in rows]

    @staticmethod
    def _row_to_message(row: Any) -> dict[str, Any]:
        """Convert a messages table row tuple to a dict."""
        tool_calls_raw = row[5]
        tool_calls = json.loads(tool_calls_raw) if tool_calls_raw else None
        return {
            "id": row[0],
            "agent_name": row[1],
            "thread_id": row[2],
            "role": row[3],
            "content": row[4],
            "tool_calls": tool_calls,
            "tool_call_id": row[6],
            "timestamp": row[7],
            "discord_message_id": row[8],
        }

    # ── Agent channel update ────────────────────────────────────────────

    async def update_agent_channel(self, name: str, new_channel_id: int) -> None:
        """Update the channel_id for an agent."""
        await self.connection.execute(
            "UPDATE agents SET channel_id = ? WHERE name = ?",
            (new_channel_id, name),
        )
        await self.connection.commit()
        logger.info("Updated agent %s channel_id to %d", name, new_channel_id)

    async def update_agent_field(self, name: str, field: str, value: str | None) -> None:
        """Update a single column in the agents table.

        Only ``permissions`` and ``model`` are allowed to prevent misuse.
        """
        allowed = {"permissions", "model"}
        if field not in allowed:
            raise ValueError(f"Cannot update field {field!r}; allowed: {allowed}")
        await self.connection.execute(
            f"UPDATE agents SET {field} = ? WHERE name = ?",  # noqa: S608
            (value, name),
        )
        await self.connection.commit()
        logger.info("Updated agent %s %s to %s", name, field, value)

    # ── Clear time ───────────────────────────────────────────────────────

    async def get_last_clear_time(self, agent_name: str) -> str | None:
        """Get the last_clear_time for an agent, or None."""
        async with self.connection.execute(
            "SELECT last_clear_time FROM agents WHERE name = ?",
            (agent_name,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return row[0]  # type: ignore[no-any-return]

    async def set_last_clear_time(self, agent_name: str, clear_time: str) -> None:
        """Set the last_clear_time for an agent."""
        await self.connection.execute(
            "UPDATE agents SET last_clear_time = ? WHERE name = ?",
            (clear_time, agent_name),
        )
        await self.connection.commit()

    # ── Session snapshot storage ─────────────────────────────────────────

    async def save_session(
        self,
        session_id: str,
        agent_name: str,
        description: str,
        summary: str,
        saved_at: str,
        message_count: int,
        file_path: str,
        window_start: str,
        window_end: str,
    ) -> None:
        """Insert a session snapshot row."""
        await self.connection.execute(
            "INSERT INTO sessions "
            "(id, agent_name, description, summary, saved_at, message_count, "
            "file_path, window_start, window_end) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                agent_name,
                description,
                summary,
                saved_at,
                message_count,
                file_path,
                window_start,
                window_end,
            ),
        )
        await self.connection.commit()

    async def list_sessions(
        self, agent_name: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """List session snapshots for an agent, most recent first."""
        async with self.connection.execute(
            "SELECT id, agent_name, description, summary, saved_at, message_count, "
            "file_path, window_start, window_end "
            "FROM sessions WHERE agent_name = ? ORDER BY saved_at DESC LIMIT ?",
            (agent_name, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                "session_id": row[0],
                "agent_name": row[1],
                "description": row[2],
                "summary": row[3],
                "saved_at": row[4],
                "message_count": row[5],
                "file_path": row[6],
                "window_start": row[7],
                "window_end": row[8],
            }
            for row in rows
        ]

    # ── Self-edit audit logging ────────────────────────────────────────

    async def log_self_edit(
        self,
        agent_name: str,
        edit_type: str,
        old_value: str,
        new_value: str,
        user_id: int | None = None,
    ) -> None:
        """Log a self-edit operation to the audit table."""
        from datetime import UTC, datetime

        detail = json.dumps({
            "edit_type": edit_type,
            "old_value": old_value[:500],
            "new_value": new_value[:500],
        })
        await self.connection.execute(
            "INSERT INTO audit_log "
            "(agent_name, timestamp, action_string, decision, user_id, detail) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                agent_name,
                datetime.now(UTC).isoformat(),
                f"tool:self_edit:{edit_type}",
                "allow",
                user_id,
                detail,
            ),
        )
        await self.connection.commit()

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Fetch a single session by ID, or None."""
        async with self.connection.execute(
            "SELECT id, agent_name, description, summary, saved_at, message_count, "
            "file_path, window_start, window_end "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "session_id": row[0],
            "agent_name": row[1],
            "description": row[2],
            "summary": row[3],
            "saved_at": row[4],
            "message_count": row[5],
            "file_path": row[6],
            "window_start": row[7],
            "window_end": row[8],
        }
