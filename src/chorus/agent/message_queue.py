"""Per-channel rate-limited message queue for Discord's 5msg/5s limit."""

from __future__ import annotations

import asyncio
import time
from typing import Any


class ChannelMessageQueue:
    """Rate-limited message sender for a single Discord channel.

    Respects Discord's rate limit of max_per_window messages per window_seconds.
    Messages are sent FIFO with async locking to prevent burst violations.
    """

    def __init__(
        self,
        channel: Any,
        max_per_window: int = 5,
        window_seconds: float = 5.0,
    ) -> None:
        self._channel = channel
        self._max_per_window = max_per_window
        self._window_seconds = window_seconds
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def send(self, content: str, **kwargs: Any) -> Any:
        """Send a message, waiting if rate limit window is full."""
        async with self._lock:
            now = time.monotonic()
            # Prune old timestamps outside the window
            self._timestamps = [
                ts for ts in self._timestamps if now - ts < self._window_seconds
            ]
            # If at capacity, wait for the oldest to expire
            if len(self._timestamps) >= self._max_per_window:
                wait_time = self._window_seconds - (now - self._timestamps[0])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    now = time.monotonic()
                    self._timestamps = [
                        ts for ts in self._timestamps if now - ts < self._window_seconds
                    ]

            self._timestamps.append(time.monotonic())
            return await self._channel.send(content, **kwargs)
