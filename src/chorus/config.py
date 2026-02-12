"""Bot configuration loaded from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("chorus.config")


@dataclass(frozen=True)
class BotConfig:
    """Immutable bot configuration. Construct via ``from_env()`` or directly for tests."""

    discord_token: str
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    chorus_home: Path = field(default_factory=lambda: Path.home() / ".chorus-agents")
    dev_guild_id: int | None = None

    @classmethod
    def from_env(cls) -> BotConfig:
        """Build config from ``os.environ``. Raises ``ValueError`` on missing token."""
        raw_token = os.environ.get("DISCORD_TOKEN", "")
        token = raw_token.strip()
        if not token:
            raise ValueError("DISCORD_TOKEN is required but missing or empty")

        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip() or None

        raw_home = os.environ.get("CHORUS_HOME", "")
        chorus_home = (
            Path(raw_home).expanduser().resolve() if raw_home else Path.home() / ".chorus-agents"
        )

        raw_guild = os.environ.get("DEV_GUILD_ID", "").strip()
        dev_guild_id = int(raw_guild) if raw_guild else None

        config = cls(
            discord_token=token,
            anthropic_api_key=anthropic_key,
            openai_api_key=openai_key,
            chorus_home=chorus_home,
            dev_guild_id=dev_guild_id,
        )
        logger.info("Config loaded — chorus_home=%s, dev_guild=%s", chorus_home, dev_guild_id)
        return config
