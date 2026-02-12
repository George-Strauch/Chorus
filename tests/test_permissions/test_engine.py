"""Tests for chorus.permissions.engine — regex-based permission matching."""

import pytest


class TestActionStringFormatting:
    def test_tool_action_format(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")

    def test_bash_action_includes_command(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")

    def test_git_action_includes_operation(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")

    def test_self_edit_action_includes_target(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")


class TestPermissionMatching:
    def test_allow_pattern_matches(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")

    def test_ask_pattern_matches(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")

    def test_no_match_denies(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")

    def test_allow_checked_before_ask(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")


class TestPresets:
    def test_open_allows_everything(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")

    def test_standard_allows_file_ops(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")

    def test_standard_asks_for_bash(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")

    def test_locked_denies_writes(self) -> None:
        pytest.skip("Not implemented yet — TODO 003")
