"""Shared test fixtures for Chorus."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from chorus.config import BotConfig


@pytest.fixture
def tmp_chorus_home(tmp_path: Path) -> Path:
    """Create a temporary ~/.chorus-agents/ structure."""
    home = tmp_path / "chorus-agents"
    home.mkdir()
    (home / "agents").mkdir()
    (home / "db").mkdir()
    return home


@pytest.fixture
def tmp_env(monkeypatch: pytest.MonkeyPatch, tmp_chorus_home: Path) -> BotConfig:
    """Set environment variables for config tests and return the expected config."""
    monkeypatch.setenv("DISCORD_TOKEN", "test-token-123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("CHORUS_HOME", str(tmp_chorus_home))
    monkeypatch.setenv("DEV_GUILD_ID", "999888777")
    return BotConfig(
        discord_token="test-token-123",
        anthropic_api_key="sk-ant-test",
        openai_api_key="sk-openai-test",
        chorus_home=tmp_chorus_home,
        dev_guild_id=999888777,
    )


@pytest.fixture
def tmp_template(tmp_path: Path) -> Path:
    """Create a temporary template directory matching the repo template/."""
    template = tmp_path / "template"
    template.mkdir()

    agent_config = {
        "name": "",
        "channel_id": None,
        "model": None,
        "system_prompt": "You are a general-purpose AI agent.",
        "permissions": "standard",
        "created_at": None,
        "running_tasks": [],
    }
    (template / "agent.json").write_text(json.dumps(agent_config, indent=4))

    docs = template / "docs"
    docs.mkdir()
    (docs / "README.md").write_text("# Agent Documentation\n")

    workspace = template / "workspace"
    workspace.mkdir()
    (workspace / ".gitkeep").touch()

    return template


@pytest.fixture
def tmp_agent_dir(tmp_chorus_home: Path, tmp_template: Path) -> Path:
    """Create a temporary agent directory by copying the template."""
    agent_dir = tmp_chorus_home / "agents" / "test-agent"
    shutil.copytree(tmp_template, agent_dir)

    # Update agent.json with test values
    config_path = agent_dir / "agent.json"
    config = json.loads(config_path.read_text())
    config["name"] = "test-agent"
    config_path.write_text(json.dumps(config, indent=4))

    # Create sessions directory
    (agent_dir / "sessions").mkdir()

    return agent_dir


@pytest.fixture
def mock_discord_ctx() -> MagicMock:
    """Create a mock Discord interaction context."""
    ctx = MagicMock()
    ctx.response = AsyncMock()
    ctx.followup = AsyncMock()
    ctx.guild = MagicMock()
    ctx.guild.id = 123456789
    ctx.guild.name = "Test Server"
    ctx.guild.create_text_channel = AsyncMock()
    ctx.channel = MagicMock()
    ctx.channel.id = 987654321
    ctx.channel.name = "test-agent"
    ctx.channel.send = AsyncMock()
    ctx.user = MagicMock()
    ctx.user.id = 111222333
    ctx.user.name = "testuser"
    ctx.user.guild_permissions = MagicMock()
    ctx.user.guild_permissions.administrator = False
    return ctx


@pytest.fixture
def mock_bot(tmp_chorus_home: Path) -> MagicMock:
    """Create a mock Discord bot instance."""
    config = BotConfig(
        discord_token="test-token",
        chorus_home=tmp_chorus_home,
    )
    bot = MagicMock()
    bot.config = config
    bot.user = MagicMock()
    bot.user.id = 444555666
    bot.latency = 0.042
    bot.guilds = []
    bot.tree = MagicMock()
    bot.tree.sync = AsyncMock()
    return bot


@pytest.fixture
def sample_permission_profiles() -> dict[str, dict[str, list[str]]]:
    """Return the built-in permission profile presets."""
    return {
        "open": {
            "allow": [".*"],
            "ask": [],
        },
        "standard": {
            "allow": [
                r"tool:(create_file|str_replace|view):.*",
                r"tool:git:(status|add|commit|diff|log|branch|checkout).*",
                r"tool:self_edit:docs/.*",
            ],
            "ask": [
                r"tool:bash:.*",
                r"tool:git:(push|merge_request).*",
                r"tool:self_edit:(system_prompt|permissions|model)",
            ],
        },
        "locked": {
            "allow": [
                r"tool:view:.*",
                r"tool:git:(status|log|diff).*",
            ],
            "ask": [],
        },
    }
