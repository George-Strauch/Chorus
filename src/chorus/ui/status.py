"""Live status feedback — plain text messages, response formatting, presence manager."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import discord

from chorus.llm.providers import Usage

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("chorus.ui.status")

# ---------------------------------------------------------------------------
# StatusSnapshot (retained for internal tracking)
# ---------------------------------------------------------------------------


@dataclass
class StatusSnapshot:
    """Point-in-time snapshot of a running thread's status."""

    agent_name: str
    thread_id: int
    status: str  # "processing", "waiting", "completed", "error", "cancelled"
    step_number: int = 0
    current_step: str = "Starting"
    active_thread_count: int = 1
    token_usage: Usage = field(default_factory=lambda: Usage(0, 0))
    llm_iterations: int = 0
    tool_calls_made: int = 0
    tools_used: list[str] = field(default_factory=list)
    recent_commands: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    error_message: str | None = None
    response_content: str | None = None


# ---------------------------------------------------------------------------
# Plain-text formatting helpers
# ---------------------------------------------------------------------------


def format_response_footer(snapshot: StatusSnapshot) -> str:
    """Build the italic footer line for a completed response.

    Format: *branch #N · X steps · 1,234 in / 567 out · $0.042 · 12.5s*
    """
    elapsed_s = snapshot.elapsed_ms / 1000
    tok_in = f"{snapshot.token_usage.input_tokens:,}"
    tok_out = f"{snapshot.token_usage.output_tokens:,}"
    cached = snapshot.token_usage.cache_read_input_tokens
    if cached > 0:
        tok_in += f" ({cached:,} cached)"

    parts = [
        f"branch #{snapshot.thread_id}",
        f"{snapshot.step_number} steps",
        f"{tok_in} in / {tok_out} out",
    ]
    if snapshot.token_usage.cost_usd > 0:
        parts.append(f"${snapshot.token_usage.cost_usd:.3f}")
    parts.append(f"{elapsed_s:.1f}s")
    return "*" + " \u00b7 ".join(parts) + "*"


_MAX_CHUNK = 1900
_MAX_CHUNKS = 10


def _find_split_point(text: str, max_len: int) -> int:
    """Find the best split point within *text* before *max_len*.

    Priority: paragraph break (\\n\\n) > line break (\\n) > sentence end ('. ')
    > hard cut.  If the candidate split would break an open code fence
    (odd number of ``` before the point), back up before the opening fence.
    """
    # Try paragraph break
    idx = text.rfind("\n\n", 0, max_len)
    if idx > 0:
        split = idx + 1  # keep one newline on left side
    else:
        # Try line break
        idx = text.rfind("\n", 0, max_len)
        if idx > 0:
            split = idx + 1
        else:
            # Try sentence end
            idx = text.rfind(". ", 0, max_len)
            split = idx + 2 if idx > 0 else max_len

    # Code-block protection: if odd number of ``` before split, back up
    before = text[:split]
    if before.count("```") % 2 == 1:
        fence_start = before.rfind("```")
        if fence_start > 0:
            split = fence_start

    return split


def chunk_response(content: str, max_chunk: int = _MAX_CHUNK) -> list[str]:
    """Split *content* into Discord-safe chunks."""
    if len(content) <= max_chunk:
        return [content]

    chunks: list[str] = []
    remaining = content
    while remaining:
        if len(chunks) >= _MAX_CHUNKS - 1:
            # Last allowed chunk — include everything remaining (may be truncated below)
            chunks.append(remaining)
            break
        if len(remaining) <= max_chunk:
            chunks.append(remaining)
            break
        split_at = _find_split_point(remaining, max_chunk)
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()

    # Truncate if final chunk is still too long after max chunks reached
    if len(chunks) == _MAX_CHUNKS and len(chunks[-1]) > max_chunk:
        omitted = len(chunks[-1]) - max_chunk + 60  # room for notice
        chunks[-1] = (
            chunks[-1][:max_chunk - 60]
            + f"\n\n*(response truncated — {omitted:,} characters omitted)*"
        )

    return chunks


# ---------------------------------------------------------------------------
# KillBranchView — red "Kill" button on the heartbeat message
# ---------------------------------------------------------------------------


class KillBranchView(discord.ui.View):
    """A single red button that kills the running branch."""

    def __init__(
        self,
        kill_callback: Callable[[], Any],
        timeout: float = 600,
    ) -> None:
        super().__init__(timeout=timeout)
        self._kill_callback = kill_callback

    @discord.ui.button(label="Kill", style=discord.ButtonStyle.danger)
    async def kill_button(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button[KillBranchView],
    ) -> None:
        button.disabled = True
        await interaction.response.edit_message(view=self)
        await self._kill_callback()

    async def on_timeout(self) -> None:
        self.disable()

    def disable(self) -> None:
        for child in self.children:
            child.disabled = True  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# LiveStatusView — plain text two-phase pattern
# ---------------------------------------------------------------------------

_THINKING_MSG = "Thinking..."
_RESPONSE_EDIT_THRESHOLD_S = 15.0
_TICKER_INTERVAL_S = 1.1


_CMD_TRUNCATE_LEN = 80


def format_status_line(snapshot: StatusSnapshot, elapsed_s: float) -> str:
    """Build a live status line shown while processing.

    Format: *Thinking (call 2) · 3.2s · 1,234 in / 567 out · 5 calls*

    When recent commands exist, appends them below the status line.
    """
    parts = [snapshot.current_step]
    parts.append(f"{elapsed_s:.1f}s")
    tok_in = f"{snapshot.token_usage.input_tokens:,}"
    tok_out = f"{snapshot.token_usage.output_tokens:,}"
    cached = snapshot.token_usage.cache_read_input_tokens
    if cached > 0:
        tok_in += f" ({cached:,} cached)"
    parts.append(f"{tok_in} in / {tok_out} out")
    if snapshot.token_usage.cost_usd > 0:
        parts.append(f"${snapshot.token_usage.cost_usd:.3f}")
    if snapshot.tool_calls_made > 0:
        n = snapshot.tool_calls_made
        parts.append(f"{n} call{'s' if n != 1 else ''}")
    line = "*" + " \u00b7 ".join(parts) + "*"

    if snapshot.recent_commands:
        cmd_lines = []
        for cmd in snapshot.recent_commands:
            if len(cmd) > _CMD_TRUNCATE_LEN:
                cmd = cmd[:_CMD_TRUNCATE_LEN] + "\u2026"
            cmd_lines.append(f"`{cmd}`")
        line += "\n\n**Recent commands:**\n" + "\n".join(cmd_lines)

    return line


class LiveStatusView:
    """Manages a plain-text status message for one thread.

    Phase 1: Sends a "Thinking..." message on ``start()`` and begins a
    background ticker that edits the message every 1.1s with live metrics
    (elapsed time, token usage, tool call count).

    Phase 2 (``finalize()``):
      - Stops the ticker.
      - If <15s elapsed: edits the message with the response.
      - If >=15s elapsed: sends a NEW message with the response.
    """

    def __init__(
        self,
        channel: Any,
        agent_name: str,
        thread_id: int,
        get_active_count: Callable[[], int],
        reference: discord.Message | None = None,
        _rate_limiter: Any | None = None,  # kept for API compat, unused
        _clock: Callable[[], float] | None = None,
        kill_callback: Callable[[], Any] | None = None,
    ) -> None:
        self._channel = channel
        self._snapshot = StatusSnapshot(
            agent_name=agent_name,
            thread_id=thread_id,
            status="processing",
        )
        self._get_active_count = get_active_count
        self._reference = reference
        self._clock = _clock or self._default_clock
        self._kill_callback = kill_callback
        self._kill_view: KillBranchView | None = None
        self._message: discord.Message | None = None
        self._started_at: float | None = None
        self._ticker_task: asyncio.Task[None] | None = None
        self._chunk_message_ids: list[int] = []

    @staticmethod
    def _default_clock() -> float:
        return asyncio.get_event_loop().time()

    @property
    def message(self) -> discord.Message | None:
        """The underlying Discord message (None until start() succeeds)."""
        return self._message

    @property
    def chunk_message_ids(self) -> list[int]:
        """All Discord message IDs from a chunked response."""
        return list(self._chunk_message_ids)

    async def start(self) -> None:
        """Send the initial 'Thinking...' message and start the ticker."""
        self._started_at = self._clock()
        try:
            kwargs: dict[str, Any] = {"content": _THINKING_MSG}
            if self._reference is not None:
                kwargs["reference"] = self._reference
            if self._kill_callback is not None:
                self._kill_view = KillBranchView(self._kill_callback)
                kwargs["view"] = self._kill_view
            self._message = await self._channel.send(**kwargs)
        except Exception:
            logger.warning("Failed to send thinking message", exc_info=True)
            return

        # Start background ticker
        self._ticker_task = asyncio.create_task(self._ticker_loop())

    async def _ticker_loop(self) -> None:
        """Edit the status message every _TICKER_INTERVAL_S seconds."""
        while True:
            await asyncio.sleep(_TICKER_INTERVAL_S)
            if self._message is None:
                break
            elapsed_s = self._elapsed_seconds()
            line = format_status_line(self._snapshot, elapsed_s)
            try:
                await self._message.edit(content=line)
            except Exception:
                logger.debug("Ticker edit failed", exc_info=True)

    def _elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return self._clock() - self._started_at

    async def update(self, **changes: Any) -> None:
        """Accept status updates — the ticker will pick them up."""
        for key, value in changes.items():
            if hasattr(self._snapshot, key):
                setattr(self._snapshot, key, value)
        self._snapshot.active_thread_count = self._get_active_count()

    def _stop_ticker(self) -> None:
        """Cancel the background ticker task."""
        if self._ticker_task is not None and not self._ticker_task.done():
            self._ticker_task.cancel()
            self._ticker_task = None

    async def finalize(
        self,
        status: str,
        error: str | None = None,
        response_content: str | None = None,
    ) -> None:
        """Stop the ticker and deliver the final response.

        Long responses are split into multiple Discord messages.
        The footer is appended only to the last chunk.

        - <15s and single chunk: edit the message in-place.
        - Otherwise: send new message(s).
        """
        self._stop_ticker()
        self._chunk_message_ids = []

        # Disable the kill button now that the branch is done
        if self._kill_view is not None:
            self._kill_view.disable()

        self._snapshot.status = status
        if error:
            self._snapshot.error_message = error
        if response_content is not None:
            self._snapshot.response_content = response_content
        self._snapshot.active_thread_count = self._get_active_count()

        # Compute elapsed
        if self._started_at is not None:
            elapsed = self._clock() - self._started_at
            self._snapshot.elapsed_ms = int(elapsed * 1000)

        # Build content chunks
        body = self._build_response_body()
        footer = format_response_footer(self._snapshot)
        chunks = chunk_response(body)

        # Append footer to the last chunk
        last_with_footer = chunks[-1] + "\n" + footer
        if len(last_with_footer) <= _MAX_CHUNK + 200:  # footer is ~100 chars
            chunks[-1] = last_with_footer
        else:
            chunks.append(footer)

        # Decide: edit-in-place or new message for first chunk
        elapsed_s = (
            (self._snapshot.elapsed_ms / 1000)
            if self._snapshot.elapsed_ms
            else 0
        )
        first_chunk = chunks[0]
        is_quick_single = (
            elapsed_s < _RESPONSE_EDIT_THRESHOLD_S
            and self._message is not None
            and len(chunks) == 1
        )
        if is_quick_single:
            # Single chunk, fast response — edit in-place
            try:
                edit_kwargs: dict[str, Any] = {"content": first_chunk, "embed": None}
                if self._kill_view is not None:
                    edit_kwargs["view"] = self._kill_view
                await self._message.edit(**edit_kwargs)
                self._chunk_message_ids.append(self._message.id)
            except Exception:
                logger.warning("Failed to edit response message", exc_info=True)
        else:
            # Send all chunks as new messages
            for chunk in chunks:
                try:
                    kwargs: dict[str, Any] = {"content": chunk}
                    if self._reference is not None and not self._chunk_message_ids:
                        kwargs["reference"] = self._reference
                    new_msg = await self._channel.send(**kwargs)
                    self._message = new_msg
                    self._chunk_message_ids.append(new_msg.id)
                except Exception:
                    logger.warning("Failed to send response chunk", exc_info=True)

    def _build_response_body(self) -> str:
        """Build the response body without footer (chunking happens in finalize)."""
        snap = self._snapshot
        parts: list[str] = []

        if snap.error_message:
            parts.append(f"**Error:** {snap.error_message}")

        if snap.response_content:
            parts.append(snap.response_content)

        if not parts:
            parts.append("*(no response)*")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# BotPresenceManager — debounced bot activity status
# ---------------------------------------------------------------------------


class BotPresenceManager:
    """Manages the bot's Discord presence based on active thread count.

    Tracks active threads across all agents and updates the bot's
    activity status.  Updates are debounced to avoid rate-limiting.
    """

    def __init__(
        self,
        bot: Any,
        debounce_seconds: float = 5.0,
        _clock: Callable[[], float] | None = None,
    ) -> None:
        self._bot = bot
        self._debounce = debounce_seconds
        self._clock = _clock or self._default_clock
        self._active: dict[str, set[int]] = {}
        self._last_update_time: float = 0.0
        self._deferred_task: asyncio.Task[None] | None = None

    @staticmethod
    def _default_clock() -> float:
        return asyncio.get_event_loop().time()

    async def thread_started(self, agent_name: str, thread_id: int) -> None:
        """Register a newly started thread."""
        self._active.setdefault(agent_name, set()).add(thread_id)
        await self._schedule_update()

    async def thread_completed(self, agent_name: str, thread_id: int) -> None:
        """Remove a completed thread."""
        if agent_name in self._active:
            self._active[agent_name].discard(thread_id)
            if not self._active[agent_name]:
                del self._active[agent_name]
        await self._schedule_update()

    async def _schedule_update(self) -> None:
        """Schedule a debounced presence update."""
        now = self._clock()
        elapsed = now - self._last_update_time
        if elapsed >= self._debounce:
            await self._do_update()
        else:
            if self._deferred_task is None or self._deferred_task.done():
                delay = self._debounce - elapsed
                self._deferred_task = asyncio.create_task(self._deferred_update(delay))

    async def _do_update(self) -> None:
        """Perform the actual presence change."""
        total_tasks = sum(len(ids) for ids in self._active.values())
        num_agents = len(self._active)

        if total_tasks > 0:
            name = f"Processing {total_tasks} task(s) | {num_agents} agent(s)"
        else:
            name = "Idle"

        activity = discord.Activity(type=discord.ActivityType.custom, name=name)
        try:
            await self._bot.change_presence(activity=activity)
            self._last_update_time = self._clock()
        except Exception:
            logger.warning("Failed to update bot presence", exc_info=True)

    async def _deferred_update(self, delay: float) -> None:
        """Wait then update presence."""
        await asyncio.sleep(delay)
        await self._do_update()
