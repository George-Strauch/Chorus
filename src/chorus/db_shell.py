"""Quick CLI for querying the Chorus SQLite database from the host.

Usage:
    python -m chorus.db_shell agents
    python -m chorus.db_shell messages eve --last 20
    python -m chorus.db_shell messages eve --search "error"
    python -m chorus.db_shell status
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def _resolve_db_path() -> Path:
    """Find the chorus DB path from environment or conventions."""
    import os

    # 1. CHORUS_HOME env var
    chorus_home = os.environ.get("CHORUS_HOME", "").strip()
    if chorus_home:
        return Path(chorus_home).expanduser().resolve() / "db" / "chorus.db"

    # 2. ./data exists in cwd (Docker host convention)
    data_dir = Path.cwd() / "data"
    if data_dir.is_dir():
        return data_dir / "db" / "chorus.db"

    # 3. Default ~/.chorus-agents
    return Path.home() / ".chorus-agents" / "db" / "chorus.db"


def cmd_agents(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    """List all agents."""
    cursor = conn.execute(
        "SELECT name, channel_id, guild_id, status FROM agents ORDER BY name"
    )
    rows = cursor.fetchall()
    if not rows:
        print("No agents found.")
        return
    print(f"{'Name':<25} {'Channel ID':<22} {'Guild ID':<22} {'Status':<10}")
    print("-" * 79)
    for name, channel_id, guild_id, status in rows:
        print(f"{name:<25} {channel_id:<22} {guild_id:<22} {status:<10}")
    print(f"\n{len(rows)} agent(s)")


def cmd_messages(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    """Show recent messages for an agent."""
    agent_name = args.agent_name
    limit = args.last
    search = args.search

    # Verify agent exists
    cursor = conn.execute("SELECT 1 FROM agents WHERE name = ?", (agent_name,))
    if cursor.fetchone() is None:
        print(f"Agent '{agent_name}' not found.")
        sys.exit(1)

    if search:
        cursor = conn.execute(
            "SELECT role, content, timestamp, thread_id FROM messages "
            "WHERE agent_name = ? AND content LIKE ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (agent_name, f"%{search}%", limit),
        )
    else:
        cursor = conn.execute(
            "SELECT role, content, timestamp, thread_id FROM messages "
            "WHERE agent_name = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (agent_name, limit),
        )

    rows = cursor.fetchall()
    if not rows:
        print(f"No messages found for agent '{agent_name}'.")
        return

    # Reverse to show oldest first
    rows = list(reversed(rows))
    for role, content, timestamp, thread_id in rows:
        thread_str = f" [thread={thread_id}]" if thread_id else ""
        content_preview = (content or "")[:200]
        if content and len(content) > 200:
            content_preview += "..."
        print(f"[{timestamp}] {role}{thread_str}: {content_preview}")
    print(f"\n{len(rows)} message(s)")


def cmd_status(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    """Show DB stats."""
    agent_count = conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
    message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    step_count = conn.execute("SELECT COUNT(*) FROM thread_steps").fetchone()[0]

    print("Chorus DB Status")
    print("-" * 30)
    print(f"Agents:        {agent_count}")
    print(f"Messages:      {message_count}")
    print(f"Sessions:      {session_count}")
    print(f"Thread steps:  {step_count}")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="chorus.db_shell",
        description="Query the Chorus SQLite database",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to chorus.db (auto-detected if not set)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # agents
    subparsers.add_parser("agents", help="List all agents")

    # messages
    msg_parser = subparsers.add_parser("messages", help="Show messages for an agent")
    msg_parser.add_argument("agent_name", help="Agent name")
    msg_parser.add_argument("--last", type=int, default=20, help="Number of messages (default: 20)")
    msg_parser.add_argument("--search", type=str, default=None, help="Search term")

    # status
    subparsers.add_parser("status", help="Show DB stats")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the db_shell CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    db_path = Path(args.db) if args.db else _resolve_db_path()
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    try:
        dispatch = {
            "agents": cmd_agents,
            "messages": cmd_messages,
            "status": cmd_status,
        }
        dispatch[args.command](conn, args)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
