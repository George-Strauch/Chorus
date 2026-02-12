"""Async SQLite storage layer for Chorus."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL REFERENCES agents(name),
    description TEXT,
    saved_at TEXT NOT NULL,
    message_count INTEGER,
    file_path TEXT NOT NULL
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
"""


class Database:
    """Async SQLite database wrapper for Chorus state."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None

    async def init(self) -> None:
        """Create parent directories, open connection, and run schema."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path)
        await self._conn.executescript(SCHEMA_SQL)
        await self._conn.commit()
        logger.info("Database initialized at %s", self._path)

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
