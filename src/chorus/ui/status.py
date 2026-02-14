"""Live status feedback — embed builder, throttled editor, presence manager."""

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
# StatusSnapshot + build_status_embed
# ---------------------------------------------------------------------------

_STATUS_COLOURS: dict[str, discord.Colour] = {
    "processing": discord.Colour.blue(),
    "waiting": discord.Colour.blue(),
    "completed": discord.Colour.blue(),
    "error": discord.Colour.red(),
    "cancelled": discord.Colour.red(),
}


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
    elapsed_ms: int = 0
    error_message: str | None = None
    response_content: str | None = None


def build_status_embed(snapshot: StatusSnapshot) -> discord.Embed:
    """Build a compact Discord embed from a status snapshot.

    Two modes:
    - **In-progress** (no response_content): title = "agent · branch #N",
      description = status + metrics.
    - **Finalized with response** (response_content set): description = response
      text (truncated ~4000 chars), footer = "branch #N · X steps · tokens · elapsed",
      title = just agent name.
    """
    colour = _STATUS_COLOURS.get(snapshot.status, discord.Colour.greyple())
    elapsed_s = snapshot.elapsed_ms / 1000
    tok_in = f"{snapshot.token_usage.input_tokens:,}"
    tok_out = f"{snapshot.token_usage.output_tokens:,}"

    # ── Finalized with response ──────────────────────────────────────
    if snapshot.response_content is not None:
        # Truncate response to ~4000 chars (Discord embed description limit is 4096)
        max_len = 4000
        content = snapshot.response_content
        if len(content) > max_len:
            content = content[:max_len] + "…"

        footer_parts = [
            f"branch #{snapshot.thread_id}",
            f"{snapshot.step_number} steps",
            f"{tok_in} in / {tok_out} out",
            f"{elapsed_s:.1f}s",
        ]
        footer_text = " · ".join(footer_parts)

        embed = discord.Embed(
            title=snapshot.agent_name,
            description=content,
            colour=colour,
        )
        embed.set_footer(text=footer_text)

        if snapshot.error_message:
            embed.description = (embed.description or "") + f"\n**Error:** {snapshot.error_message}"

        return embed

    # ── In-progress (no response_content) ────────────────────────────
    status_label = snapshot.status.capitalize()

    # Line 1: status + step info
    if snapshot.status in ("completed", "error", "cancelled"):
        line1 = f"**{status_label}** · {snapshot.step_number} steps"
    elif snapshot.step_number > 0:
        line1 = f"**{status_label}** · Step {snapshot.step_number}: {snapshot.current_step}"
    else:
        line1 = f"**{status_label}** · {snapshot.current_step}"

    # Line 2: metrics
    parts = [f"{tok_in} in / {tok_out} out"]
    if snapshot.llm_iterations:
        calls = "call" if snapshot.llm_iterations == 1 else "calls"
        parts.append(f"{snapshot.llm_iterations} {calls}")
    if snapshot.tool_calls_made:
        parts.append(f"{snapshot.tool_calls_made} tools")
    parts.append(f"{elapsed_s:.1f}s")
    line2 = " · ".join(parts)

    description = f"{line1}\n{line2}"

    if snapshot.error_message:
        description += f"\n**Error:** {snapshot.error_message}"

    return discord.Embed(
        title=f"{snapshot.agent_name} · branch #{snapshot.thread_id}",
        description=description,
        colour=colour,
    )


# ---------------------------------------------------------------------------
# GlobalEditRateLimiter
# ---------------------------------------------------------------------------


class GlobalEditRateLimiter:
    """Global rate limiter for Discord message edits across all LiveStatusViews.

    No asyncio.Lock needed — single-threaded event loop.
    """

    def __init__(
        self,
        min_interval: float = 1.1,
        _clock: Callable[[], float] | None = None,
    ) -> None:
        self._min_interval = min_interval
        self._clock = _clock or self._default_clock
        self._last_edit_time: float = float("-inf")

    @staticmethod
    def _default_clock() -> float:
        return asyncio.get_event_loop().time()

    def can_edit_now(self) -> bool:
        """Return True if enough time has passed since the last recorded edit."""
        now = self._clock()
        return (now - self._last_edit_time) >= self._min_interval

    def time_until_next_allowed(self) -> float:
        """Seconds remaining until the next edit is allowed (0.0 if allowed now)."""
        now = self._clock()
        remaining = self._min_interval - (now - self._last_edit_time)
        return max(0.0, remaining)

    def record_edit(self) -> None:
        """Record that an edit was just performed."""
        self._last_edit_time = self._clock()


# Module-level default instance
_global_rate_limiter = GlobalEditRateLimiter()


# ---------------------------------------------------------------------------
# LiveStatusView — throttled embed editor
# ---------------------------------------------------------------------------


class LiveStatusView:
    """Manages a single live-updating status embed for one thread.

    Sends an initial embed on ``start()``, edits it on ``update()``,
    and does a final edit on ``finalize()``.  Edits are throttled via
    a shared ``GlobalEditRateLimiter``.
    """

    def __init__(
        self,
        channel: Any,
        agent_name: str,
        thread_id: int,
        get_active_count: Callable[[], int],
        reference: discord.Message | None = None,
        _rate_limiter: GlobalEditRateLimiter | None = None,
        _clock: Callable[[], float] | None = None,
    ) -> None:
        self._channel = channel
        self._snapshot = StatusSnapshot(
            agent_name=agent_name,
            thread_id=thread_id,
            status="processing",
        )
        self._get_active_count = get_active_count
        self._reference = reference
        self._rate_limiter = _rate_limiter or _global_rate_limiter
        self._clock = _clock or self._default_clock
        self._message: discord.Message | None = None
        self._pending = False
        self._deferred_task: asyncio.Task[None] | None = None
        self._started_at: float | None = None

    @staticmethod
    def _default_clock() -> float:
        return asyncio.get_event_loop().time()

    @property
    def message(self) -> discord.Message | None:
        """The underlying Discord message (None until start() succeeds)."""
        return self._message

    async def start(self) -> None:
        """Send the initial status embed."""
        self._started_at = self._clock()
        self._snapshot.active_thread_count = self._get_active_count()
        embed = build_status_embed(self._snapshot)
        try:
            kwargs: dict[str, Any] = {"embed": embed}
            if self._reference is not None:
                kwargs["reference"] = self._reference
            self._message = await self._channel.send(**kwargs)
            self._rate_limiter.record_edit()
        except Exception:
            logger.warning("Failed to send status embed", exc_info=True)

    async def update(self, **changes: Any) -> None:
        """Merge changes into the snapshot and schedule a throttled edit."""
        if self._message is None:
            return

        for key, value in changes.items():
            if hasattr(self._snapshot, key):
                setattr(self._snapshot, key, value)
        self._snapshot.active_thread_count = self._get_active_count()

        if self._rate_limiter.can_edit_now():
            await self._do_edit()
        else:
            # Schedule a deferred edit if not already pending
            if not self._pending:
                self._pending = True
                delay = self._rate_limiter.time_until_next_allowed()
                self._deferred_task = asyncio.create_task(self._deferred_edit(delay))

    async def finalize(
        self,
        status: str,
        error: str | None = None,
        response_content: str | None = None,
    ) -> None:
        """Final edit — bypasses rate limiter, always edits immediately."""
        if self._deferred_task is not None and not self._deferred_task.done():
            self._deferred_task.cancel()
            self._deferred_task = None
        self._pending = False

        self._snapshot.status = status
        if error:
            self._snapshot.error_message = error
        if response_content is not None:
            self._snapshot.response_content = response_content
        self._snapshot.active_thread_count = self._get_active_count()

        if self._message is not None:
            await self._do_edit()
            self._rate_limiter.record_edit()

    async def _do_edit(self) -> None:
        """Perform the actual message edit."""
        assert self._message is not None
        # Auto-compute elapsed_ms from creation time
        if self._started_at is not None:
            elapsed = self._clock() - self._started_at
            self._snapshot.elapsed_ms = int(elapsed * 1000)
        embed = build_status_embed(self._snapshot)
        try:
            await self._message.edit(embed=embed)
            self._rate_limiter.record_edit()
        except Exception:
            logger.warning("Failed to edit status embed", exc_info=True)

    async def _deferred_edit(self, delay: float) -> None:
        """Wait then edit, absorbing any updates that arrived meanwhile."""
        await asyncio.sleep(delay)
        self._pending = False
        # Re-check rate limiter — another view may have consumed the slot
        if self._message is not None and self._rate_limiter.can_edit_now():
            await self._do_edit()


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
