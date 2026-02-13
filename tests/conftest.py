"""Shared test fixtures for Chorus."""

from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

from chorus.agent.threads import ThreadManager
from chorus.config import BotConfig
from chorus.storage.db import Database


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
def workspace_dir(tmp_agent_dir: Path) -> Path:
    """Return the workspace directory inside the tmp agent, pre-populated with test files."""
    ws = tmp_agent_dir / "workspace"
    # Create a few files for testing
    (ws / "hello.txt").write_text("Hello, world!\n", encoding="utf-8")
    (ws / "src").mkdir()
    (ws / "src" / "app.py").write_text(
        "def main():\n    print('hello')\n\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )
    return ws


@pytest.fixture
def safe_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set dummy secret env vars to verify they're stripped from subprocess."""
    monkeypatch.setenv("DISCORD_TOKEN", "dummy-discord-token-for-testing")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-dummy-key")


@pytest.fixture
def git_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace with an initialized git repo and initial commit."""
    ws = tmp_path / "git-workspace"
    ws.mkdir()

    import subprocess

    env = {"HOME": str(tmp_path), "PATH": os.environ.get("PATH", "")}
    run = lambda cmd: subprocess.run(  # noqa: E731
        cmd, cwd=ws, env=env, capture_output=True, text=True, check=True
    )

    run(["git", "init", "-b", "main"])
    run(["git", "config", "user.name", "test-agent"])
    run(["git", "config", "user.email", "test-agent@chorus.local"])
    (ws / "README.md").write_text("# Test Repo\n")
    run(["git", "add", "README.md"])
    run(["git", "commit", "-m", "Initial commit"])
    return ws


@pytest.fixture
def git_workspace_with_remote(tmp_path: Path) -> tuple[Path, Path]:
    """Workspace with a bare repo added as origin, initial push done.

    Returns (workspace, bare_remote).
    """
    import subprocess

    bare = tmp_path / "remote.git"
    bare.mkdir()
    ws = tmp_path / "workspace"
    ws.mkdir()

    env = {"HOME": str(tmp_path), "PATH": os.environ.get("PATH", "")}
    run_in = lambda cwd, cmd: subprocess.run(  # noqa: E731
        cmd, cwd=cwd, env=env, capture_output=True, text=True, check=True
    )

    # Init bare remote
    run_in(bare, ["git", "init", "--bare", "-b", "main"])

    # Init workspace
    run_in(ws, ["git", "init", "-b", "main"])
    run_in(ws, ["git", "config", "user.name", "test-agent"])
    run_in(ws, ["git", "config", "user.email", "test-agent@chorus.local"])
    (ws / "README.md").write_text("# Test Repo\n")
    run_in(ws, ["git", "add", "README.md"])
    run_in(ws, ["git", "commit", "-m", "Initial commit"])
    run_in(ws, ["git", "remote", "add", "origin", str(bare)])
    run_in(ws, ["git", "push", "-u", "origin", "main"])
    return ws, bare


@pytest.fixture
def thread_manager() -> ThreadManager:
    """Create a ThreadManager for testing (no DB)."""
    return ThreadManager("test-agent")


@pytest.fixture
async def thread_db(tmp_path: Path) -> AsyncGenerator[Database, None]:
    """Create a temporary Database for thread tests."""
    db = Database(tmp_path / "db" / "chorus.db")
    await db.init()
    yield db
    await db.close()


@pytest.fixture
async def thread_manager_with_db(thread_db: Database) -> ThreadManager:
    """Create a ThreadManager with a real database."""
    return ThreadManager("test-agent", db=thread_db)


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


@pytest.fixture
async def context_db(tmp_path: Path) -> AsyncGenerator[Database, None]:
    """Create a temporary Database for context management tests."""
    db = Database(tmp_path / "db" / "chorus.db")
    await db.init()
    yield db
    await db.close()
