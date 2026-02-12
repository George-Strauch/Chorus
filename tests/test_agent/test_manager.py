"""Tests for chorus.agent.manager — agent lifecycle."""

import pytest


class TestAgentCreate:
    def test_creates_from_template(self) -> None:
        pytest.skip("Not implemented yet — TODO 002")

    def test_creates_workspace_git_repo(self) -> None:
        pytest.skip("Not implemented yet — TODO 002")

    def test_merges_config_overrides(self) -> None:
        pytest.skip("Not implemented yet — TODO 002")

    def test_duplicate_name_raises(self) -> None:
        pytest.skip("Not implemented yet — TODO 002")


class TestAgentDestroy:
    def test_removes_agent_directory(self) -> None:
        pytest.skip("Not implemented yet — TODO 002")

    def test_keep_files_option(self) -> None:
        pytest.skip("Not implemented yet — TODO 002")


class TestAgentList:
    def test_lists_active_agents(self) -> None:
        pytest.skip("Not implemented yet — TODO 002")

    def test_empty_when_no_agents(self) -> None:
        pytest.skip("Not implemented yet — TODO 002")
