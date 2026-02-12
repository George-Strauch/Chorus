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
