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

# ---------------------------------------------------------------------------
# Token estimation and model context limits
# ---------------------------------------------------------------------------

MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # Anthropic models
    "claude-opus-4-6": 200_000,
    "claude-opus-4-5-20251101": 200_000,
    "claude-opus-4-1-20250805": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-sonnet-4-5-20250929": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-haiku-4-20250506": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    # OpenAI models
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1-nano": 1_047_576,
    "gpt-5": 128_000,
    "gpt-5-mini": 128_000,
    "gpt-5-nano": 128_000,
    "gpt-5-pro": 128_000,
    "gpt-5.1": 128_000,
    "gpt-5.2": 128_000,
    "gpt-5.2-pro": 128_000,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o1-pro": 200_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o3-pro": 200_000,
    "o4-mini": 200_000,
}

# Default for unknown models
_DEFAULT_CONTEXT_LIMIT = 128_000

# Hard ceiling — Anthropic charges premium rates above 200K input tokens
MAX_INPUT_TOKENS = 200_000

# Budget is 80% of max context
_CONTEXT_BUDGET_RATIO = 0.80


def _get_context_limit(model: str | None) -> int:
    """Return the token context limit for a model, capped at MAX_INPUT_TOKENS."""
    if model is None:
        return min(_DEFAULT_CONTEXT_LIMIT, MAX_INPUT_TOKENS)
    # Exact match first
    if model in MODEL_CONTEXT_LIMITS:
        return min(MODEL_CONTEXT_LIMITS[model], MAX_INPUT_TOKENS)
    # Prefix match for dated model variants
    for prefix, limit in MODEL_CONTEXT_LIMITS.items():
        if model.startswith(prefix):
            return min(limit, MAX_INPUT_TOKENS)
    return min(_DEFAULT_CONTEXT_LIMIT, MAX_INPUT_TOKENS)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: chars / 4."""
    return len(text) // 4


def estimate_message_tokens(msg: dict[str, Any]) -> int:
    """Estimate tokens for a single message dict.

    Handles standard messages, messages with tool_calls lists,
    and messages with _anthropic_content raw blocks.
    """
    tokens = 4  # overhead for role and structure

    content = msg.get("content") or ""
    tokens += estimate_tokens(content)

    # Tool calls: estimate the arguments JSON
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            tokens += estimate_tokens(tc.get("name", ""))
            args = tc.get("arguments")
            if isinstance(args, dict):
                tokens += estimate_tokens(json.dumps(args))
            elif isinstance(args, str):
                tokens += estimate_tokens(args)

    # Anthropic raw content blocks
    raw_content = msg.get("_anthropic_content")
    if raw_content and isinstance(raw_content, list):
        tokens += estimate_tokens(json.dumps(raw_content))

    return tokens


def _truncate_to_budget(
    messages: list[dict[str, Any]], budget_tokens: int
) -> list[dict[str, Any]]:
    """Truncate the oldest messages to fit within a token budget.

    Preserves system messages at the start and keeps the most recent messages.
    """
    if not messages:
        return messages

    # Separate system messages from conversation
    system_msgs: list[dict[str, Any]] = []
    conv_msgs: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            system_msgs.append(msg)
        else:
            conv_msgs.append(msg)

    # Calculate system message overhead
    system_tokens = sum(estimate_message_tokens(m) for m in system_msgs)
    remaining_budget = budget_tokens - system_tokens

    if remaining_budget <= 0:
        # System messages alone exceed budget — return system + last message
        return system_msgs + conv_msgs[-1:] if conv_msgs else system_msgs

    # Walk backwards through conversation messages, accumulating until budget
    kept: list[dict[str, Any]] = []
    total = 0
    for msg in reversed(conv_msgs):
        msg_tokens = estimate_message_tokens(msg)
        if total + msg_tokens > remaining_budget:
            break
        kept.append(msg)
        total += msg_tokens

    kept.reverse()
    return system_msgs + kept


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
    *,
    model: str | None = None,
    available_models: list[str] | None = None,
    previous_branch_summary: str | None = None,
    previous_branch_id: int | None = None,
    scope_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Assemble the full context for an LLM call.

    Returns a list of message dicts suitable for sending to an LLM provider.

    Order:
    1. System prompt
    2. Agent docs (appended to system prompt)
    3. Model self-awareness info
    4. Scope path info (if configured)
    5. Previous branch summary (if any)
    6. Thread/process status
    7. Rolling window messages (scoped to branch, truncated to 80% budget)
    """
    messages: list[dict[str, Any]] = []

    # 1. System prompt + 2. Agent docs
    system_parts = [agent.system_prompt]
    if agent_docs_dir is not None and agent_docs_dir.is_dir():
        docs_content = _read_agent_docs(agent_docs_dir)
        if docs_content:
            system_parts.append(f"\n\n## Agent Documentation\n\n{docs_content}")

    # 3. Model self-awareness
    effective_model = model or agent.model or "unknown"
    system_parts.append(f"\n\nYou are running on model: {effective_model}.")
    if available_models:
        model_list = ", ".join(available_models[:20])  # cap to avoid bloat
        system_parts.append(f"Available models: {model_list}.")

    # 4. Scope path awareness
    if scope_path is not None:
        system_parts.append(
            f"\n\n## Host Filesystem Access\n\n"
            f"The host user's filesystem is mounted at `{scope_path}`. "
            f"You can read and write files there using absolute paths in file tools and bash commands. "
            f"The environment variable `$SCOPE_PATH` is also available in bash and expands to `{scope_path}`."
        )

    # Claude Code awareness
    from chorus.tools.claude_code import is_claude_code_available

    if is_claude_code_available():
        system_parts.append(
            "\n\n## Code Editing\n\n"
            "You have access to the `claude_code` tool for creating and editing code files "
            "(.py, .js, .ts, .go, .rs, etc.). Delegate code editing tasks to this tool for "
            "better results. For non-code files (.md, .txt, .json, .yaml), use create_file "
            "and str_replace."
        )

    # File writing guidance
    system_parts.append(
        "\n\n## File Writing\n\n"
        "When creating large files, use `append_file` in multiple tool calls to build "
        "the content incrementally. Do NOT try to write an entire large file in a single "
        "`create_file` call — the response may be cut off by output token limits. Instead: "
        "use `create_file` for the first chunk, then `append_file` for subsequent chunks."
    )

    messages.append({"role": "system", "content": "\n".join(system_parts)})

    # 5. Previous branch summary
    if previous_branch_summary and previous_branch_id is not None:
        messages.append({
            "role": "system",
            "content": f"Previous conversation (branch #{previous_branch_id}): {previous_branch_summary}",
        })

    # 6. Thread status
    current_thread_id = thread_id or 0
    thread_status = build_thread_status(thread_manager, current_thread_id)
    if thread_status and thread_status != "No active threads.":
        messages.append({"role": "system", "content": thread_status})

    # 7. Rolling window messages — scoped to this branch
    window_msgs = await context_manager.get_context(thread_id=thread_id)
    for msg in window_msgs:
        messages.append({
            "role": msg["role"],
            "content": msg.get("content", ""),
        })

    # 8. Token budget truncation (80% of model max)
    context_limit = _get_context_limit(model)
    budget = int(context_limit * _CONTEXT_BUDGET_RATIO)
    messages = _truncate_to_budget(messages, budget)

    return messages
