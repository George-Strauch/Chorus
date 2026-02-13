"""Context window management for agent conversations."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chorus.agent.threads import ThreadManager, build_thread_status
from chorus.models import Agent, SessionMetadata, SessionNotFoundError

if TYPE_CHECKING:
    from chorus.storage.db import Database

logger = logging.getLogger("chorus.agent.context")

# Type alias for the summarizer callback
Summarizer = Callable[[list[dict[str, Any]]], Coroutine[Any, Any, str]]


class ContextManager:
    """Manages the rolling context window for a single agent.

    All messages are persisted to SQLite. The context window is bounded by
    ``max(last_clear_time, now - rolling_window)``.
    """

    def __init__(
        self,
        agent_name: str,
        db: Database,
        sessions_dir: Path,
        rolling_window: int = 86400,
    ) -> None:
        self._agent_name = agent_name
        self._db = db
        self._sessions_dir = sessions_dir
        self._rolling_window = rolling_window

    # ── Message persistence ──────────────────────────────────────────────

    async def persist_message(
        self,
        *,
        role: str,
        content: str | None = None,
        thread_id: int | None = None,
        tool_calls: list[Any] | None = None,
        tool_call_id: str | None = None,
        discord_message_id: int | None = None,
    ) -> None:
        """Store a message in SQLite with auto-generated timestamp."""
        timestamp = datetime.now(UTC).isoformat()
        await self._db.persist_message(
            agent_name=self._agent_name,
            role=role,
            timestamp=timestamp,
            thread_id=thread_id,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            discord_message_id=discord_message_id,
        )

    # ── Context retrieval ────────────────────────────────────────────────

    async def get_context(
        self, thread_id: int | None = None
    ) -> list[dict[str, Any]]:
        """Return messages within the rolling window, optionally filtered by thread."""
        cutoff = await self._compute_cutoff()
        return await self._db.get_messages_since(
            self._agent_name, cutoff, thread_id=thread_id
        )

    async def _compute_cutoff(self) -> str:
        """Compute the effective cutoff timestamp for the rolling window."""
        rolling_start = (
            datetime.now(UTC) - timedelta(seconds=self._rolling_window)
        ).isoformat()

        last_clear = await self._db.get_last_clear_time(self._agent_name)
        if last_clear is None:
            return rolling_start

        # Use whichever is more recent
        return max(rolling_start, last_clear)

    # ── Clear ────────────────────────────────────────────────────────────

    async def clear(self) -> None:
        """Advance last_clear_time to now, excluding prior messages from context."""
        now = datetime.now(UTC).isoformat()
        await self._db.set_last_clear_time(self._agent_name, now)
        logger.info("Cleared context for agent %s at %s", self._agent_name, now)

    # ── Snapshot save ────────────────────────────────────────────────────

    async def save_snapshot(
        self,
        description: str = "",
        summarizer: Summarizer | None = None,
    ) -> SessionMetadata:
        """Save the current rolling window to a session JSON file.

        Does NOT clear context.
        """
        now = datetime.now(UTC)
        session_id = str(uuid.uuid4())

        # Get current window messages
        messages = await self.get_context()

        # Compute window bounds
        window_start = messages[0]["timestamp"] if messages else now.isoformat()
        window_end = messages[-1]["timestamp"] if messages else now.isoformat()

        # Generate summary
        summary = ""
        if summarizer is not None:
            try:
                summary = await summarizer(messages)
            except Exception:
                logger.exception("Summary generation failed for session %s", session_id)
                summary = "(summary generation failed)"

        # Write session file
        session_data = {
            "session_id": session_id,
            "timestamp": now.isoformat(),
            "description": description,
            "summary": summary,
            "message_count": len(messages),
            "window_start": window_start,
            "window_end": window_end,
            "messages": messages,
        }

        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._sessions_dir / f"{session_id}.json"
        file_path.write_text(json.dumps(session_data, indent=2, default=str))

        # Store metadata in DB
        await self._db.save_session(
            session_id=session_id,
            agent_name=self._agent_name,
            description=description,
            summary=summary,
            saved_at=now.isoformat(),
            message_count=len(messages),
            file_path=str(file_path),
            window_start=window_start,
            window_end=window_end,
        )

        meta = SessionMetadata(
            session_id=session_id,
            agent_name=self._agent_name,
            description=description,
            summary=summary,
            saved_at=now.isoformat(),
            message_count=len(messages),
            file_path=str(file_path),
            window_start=window_start,
            window_end=window_end,
        )
        logger.info(
            "Saved session %s for agent %s (%d messages)",
            session_id,
            self._agent_name,
            len(messages),
        )
        return meta

    # ── Snapshot list / restore ──────────────────────────────────────────

    async def list_snapshots(self, limit: int = 50) -> list[SessionMetadata]:
        """List saved session snapshots, most recent first."""
        rows = await self._db.list_sessions(self._agent_name, limit=limit)
        return [SessionMetadata.from_dict(row) for row in rows]

    async def restore_snapshot(self, session_id: str) -> None:
        """Load messages from a saved session back into the rolling window.

        Re-inserts messages with current timestamps so they appear in the
        active context window.
        """
        session = await self._db.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session {session_id!r} not found")

        file_path = Path(session["file_path"])
        if not file_path.exists():
            raise SessionNotFoundError(
                f"Session file {file_path} not found on disk"
            )

        data = json.loads(file_path.read_text())
        messages = data.get("messages", [])

        # Re-insert messages with current timestamps
        for msg in messages:
            await self.persist_message(
                role=msg["role"],
                content=msg.get("content"),
                thread_id=msg.get("thread_id"),
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
            )

        logger.info(
            "Restored session %s (%d messages) for agent %s",
            session_id,
            len(messages),
            self._agent_name,
        )


# ── Context assembly helper ─────────────────────────────────────────────


def _read_agent_docs(docs_dir: Path) -> str:
    """Read all .md files from an agent's docs/ directory."""
    parts: list[str] = []
    for md_file in sorted(docs_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if content:
            parts.append(f"--- {md_file.name} ---\n{content}")
    return "\n\n".join(parts)


async def build_llm_context(
    agent: Agent,
    thread_id: int | None,
    context_manager: ContextManager,
    thread_manager: ThreadManager,
    agent_docs_dir: Path | None,
) -> list[dict[str, Any]]:
    """Assemble the full context for an LLM call.

    Returns a list of message dicts suitable for sending to an LLM provider.

    Order:
    1. System prompt
    2. Agent docs (appended to system prompt)
    3. Thread/process status
    4. Rolling window messages
    """
    messages: list[dict[str, Any]] = []

    # 1. System prompt + 2. Agent docs
    system_parts = [agent.system_prompt]
    if agent_docs_dir is not None and agent_docs_dir.is_dir():
        docs_content = _read_agent_docs(agent_docs_dir)
        if docs_content:
            system_parts.append(f"\n\n## Agent Documentation\n\n{docs_content}")

    messages.append({"role": "system", "content": "\n".join(system_parts)})

    # 3. Thread status
    current_thread_id = thread_id or 0
    thread_status = build_thread_status(thread_manager, current_thread_id)
    if thread_status and thread_status != "No active threads.":
        messages.append({"role": "system", "content": thread_status})

    # 4. Rolling window messages
    window_msgs = await context_manager.get_context(thread_id=thread_id)
    for msg in window_msgs:
        messages.append({
            "role": msg["role"],
            "content": msg.get("content", ""),
        })

    return messages
