"""Tests for chorus.agent.directory — agent working directory management."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from chorus.agent.directory import AgentDirectory
from chorus.models import AgentExistsError

if TYPE_CHECKING:
    from pathlib import Path


class TestAgentDirectory:
    def test_ensure_home_creates_directory_structure(
        self, tmp_path: Path, tmp_template: Path
    ) -> None:
        home = tmp_path / "new-chorus-home"
        d = AgentDirectory(home, tmp_template)
        d.ensure_home()
        assert (home / "agents").is_dir()
        assert (home / "db").is_dir()

    def test_create_copies_template_to_agent_dir(
        self, tmp_chorus_home: Path, tmp_template: Path
    ) -> None:
        d = AgentDirectory(tmp_chorus_home, tmp_template)
        agent_path = d.create("test-bot")
        assert (agent_path / "agent.json").exists()
        assert (agent_path / "docs").is_dir()
        assert (agent_path / "workspace").is_dir()

    def test_create_initializes_git_repo_in_workspace(
        self, tmp_chorus_home: Path, tmp_template: Path
    ) -> None:
        d = AgentDirectory(tmp_chorus_home, tmp_template)
        agent_path = d.create("test-bot")
        assert (agent_path / "workspace" / ".git").is_dir()

    def test_create_raises_if_agent_already_exists(
        self, tmp_chorus_home: Path, tmp_template: Path
    ) -> None:
        d = AgentDirectory(tmp_chorus_home, tmp_template)
        d.create("test-bot")
        with pytest.raises(AgentExistsError):
            d.create("test-bot")

    def test_create_writes_name_and_timestamp_to_agent_json(
        self, tmp_chorus_home: Path, tmp_template: Path
    ) -> None:
        d = AgentDirectory(tmp_chorus_home, tmp_template)
        agent_path = d.create("test-bot")
        data = json.loads((agent_path / "agent.json").read_text())
        assert data["name"] == "test-bot"
        assert data["created_at"] is not None

    def test_create_applies_overrides_to_agent_json(
        self, tmp_chorus_home: Path, tmp_template: Path
    ) -> None:
        d = AgentDirectory(tmp_chorus_home, tmp_template)
        overrides = {
            "system_prompt": "Custom prompt",
            "model": "gpt-4o",
            "permissions": "open",
        }
        agent_path = d.create("test-bot", overrides=overrides)
        data = json.loads((agent_path / "agent.json").read_text())
        assert data["system_prompt"] == "Custom prompt"
        assert data["model"] == "gpt-4o"
        assert data["permissions"] == "open"

    def test_destroy_removes_agent_directory(
        self, tmp_chorus_home: Path, tmp_template: Path
    ) -> None:
        d = AgentDirectory(tmp_chorus_home, tmp_template)
        agent_path = d.create("test-bot")
        assert agent_path.exists()
        d.destroy("test-bot")
        assert not agent_path.exists()

    def test_destroy_keep_files_moves_to_trash(
        self, tmp_chorus_home: Path, tmp_template: Path
    ) -> None:
        d = AgentDirectory(tmp_chorus_home, tmp_template)
        d.create("test-bot")
        d.destroy("test-bot", keep_files=True)
        assert not (tmp_chorus_home / "agents" / "test-bot").exists()
        assert (tmp_chorus_home / ".trash" / "test-bot").exists()

    def test_get_returns_path_for_existing_agent(
        self, tmp_chorus_home: Path, tmp_template: Path
    ) -> None:
        d = AgentDirectory(tmp_chorus_home, tmp_template)
        created = d.create("test-bot")
        result = d.get("test-bot")
        assert result == created

    def test_get_returns_none_for_missing_agent(
        self, tmp_chorus_home: Path, tmp_template: Path
    ) -> None:
        d = AgentDirectory(tmp_chorus_home, tmp_template)
        assert d.get("nonexistent") is None

    def test_list_all_returns_agent_names(
        self, tmp_chorus_home: Path, tmp_template: Path
    ) -> None:
        d = AgentDirectory(tmp_chorus_home, tmp_template)
        d.create("alpha-bot")
        d.create("beta-bot")
        names = d.list_all()
        assert names == ["alpha-bot", "beta-bot"]

    def test_list_all_returns_empty_when_no_agents(
        self, tmp_chorus_home: Path, tmp_template: Path
    ) -> None:
        d = AgentDirectory(tmp_chorus_home, tmp_template)
        assert d.list_all() == []
