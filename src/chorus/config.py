"""Bot configuration loaded from environment variables."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
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
    live_test_enabled: bool = False
    live_test_channel: str | None = None

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

        raw_guild = (
            os.environ.get("DEV_GUILD_ID", "").strip()
            or os.environ.get("DISCORD_GUILD_ID", "").strip()
        )
        dev_guild_id = int(raw_guild) if raw_guild else None

        raw_live_test = os.environ.get("LIVE_TEST_ENABLED", "").strip().lower()
        live_test_enabled = raw_live_test in ("1", "true", "yes")

        raw_live_channel = os.environ.get("LIVE_TEST_CHANNEL", "").strip()
        live_test_channel = raw_live_channel if raw_live_channel else None

        config = cls(
            discord_token=token,
            anthropic_api_key=anthropic_key,
            openai_api_key=openai_key,
            chorus_home=chorus_home,
            dev_guild_id=dev_guild_id,
            live_test_enabled=live_test_enabled,
            live_test_channel=live_test_channel,
        )
        logger.info("Config loaded — chorus_home=%s, dev_guild=%s", chorus_home, dev_guild_id)
        return config


@dataclass
class GlobalConfig:
    """Server-wide default settings stored in ``config.json``."""

    default_model: str | None = None
    default_permissions: str = "standard"
    idle_timeout: int = 1800
    max_tool_loop_iterations: int = 25
    max_bash_timeout: int = 120

    @classmethod
    def load(cls, path: Path) -> GlobalConfig:
        """Load from *path*, creating a default file if it doesn't exist."""
        if path.exists():
            data = json.loads(path.read_text())
            return cls(
                default_model=data.get("default_model"),
                default_permissions=data.get("default_permissions", "standard"),
                idle_timeout=data.get("idle_timeout", 1800),
                max_tool_loop_iterations=data.get("max_tool_loop_iterations", 25),
                max_bash_timeout=data.get("max_bash_timeout", 120),
            )
        cfg = cls()
        cfg.save(path)
        return cfg

    def save(self, path: Path) -> None:
        """Persist to *path* as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def validate_model_available(model: str, chorus_home: Path) -> bool:
    """Check if *model* appears in any provider's model list in the cache."""
    from chorus.llm.discovery import read_cache

    cache = read_cache(chorus_home)
    if cache is None:
        return False
    providers = cache.get("providers", {})
    return any(model in prov.get("models", []) for prov in providers.values())
