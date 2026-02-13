"""Agent working directory management — filesystem operations for agent lifecycle."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from chorus.models import AgentExistsError

logger = logging.getLogger("chorus.agent.directory")


def read_agent_json(path: Path) -> dict[str, Any]:
    """Read and parse an agent.json file."""
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def write_agent_json(path: Path, data: dict[str, Any]) -> None:
    """Write data to an agent.json file."""
    path.write_text(json.dumps(data, indent=4) + "\n")


class AgentDirectory:
    """Manages agent directories under the chorus home."""

    def __init__(self, chorus_home: Path, template_dir: Path) -> None:
        self._home = chorus_home
        self._template = template_dir
        self._agents_dir = chorus_home / "agents"

    def ensure_home(self) -> None:
        """Create the chorus home directory structure (idempotent)."""
        self._agents_dir.mkdir(parents=True, exist_ok=True)
        (self._home / "db").mkdir(parents=True, exist_ok=True)

    def create(self, name: str, overrides: dict[str, Any] | None = None) -> Path:
        """Create a new agent directory from the template.

        Copies template, initializes git in workspace, writes name/timestamp/overrides
        to agent.json. Raises AgentExistsError if the agent directory already exists.
        """
        agent_dir = self._agents_dir / name
        if agent_dir.exists():
            raise AgentExistsError(f"Agent {name!r} already exists")

        shutil.copytree(self._template, agent_dir)

        # Create sessions directory
        (agent_dir / "sessions").mkdir(exist_ok=True)

        # Initialize git repo in workspace
        workspace = agent_dir / "workspace"
        subprocess.run(
            ["git", "init"],
            cwd=workspace,
            capture_output=True,
            check=True,
        )

        # Update agent.json with name, timestamp, and overrides
        config_path = agent_dir / "agent.json"
        data = read_agent_json(config_path)
        data["name"] = name
        data["created_at"] = datetime.now(UTC).isoformat()
        if overrides:
            for key, value in overrides.items():
                data[key] = value
        write_agent_json(config_path, data)

        logger.info("Created agent directory: %s", agent_dir)
        return agent_dir

    def destroy(self, name: str, keep_files: bool = False) -> None:
        """Remove an agent directory, or move it to .trash/ if keep_files is True."""
        agent_dir = self._agents_dir / name
        if keep_files:
            trash_dir = self._home / ".trash"
            trash_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(agent_dir), str(trash_dir / name))
            logger.info("Moved agent %s to trash", name)
        else:
            shutil.rmtree(agent_dir)
            logger.info("Destroyed agent directory: %s", name)

    def get(self, name: str) -> Path | None:
        """Return the path for an existing agent, or None if it doesn't exist."""
        agent_dir = self._agents_dir / name
        if agent_dir.is_dir():
            return agent_dir
        return None

    def update_channel_id(self, name: str, channel_id: int) -> None:
        """Update the channel_id in an agent's agent.json."""
        agent_dir = self._agents_dir / name
        config_path = agent_dir / "agent.json"
        data = read_agent_json(config_path)
        data["channel_id"] = channel_id
        write_agent_json(config_path, data)
        logger.info("Updated agent %s agent.json channel_id to %d", name, channel_id)

    def list_all(self) -> list[str]:
        """Return a sorted list of all agent names."""
        if not self._agents_dir.exists():
            return []
        return sorted(
            d.name for d in self._agents_dir.iterdir() if d.is_dir()
        )
