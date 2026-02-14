"""Tests for chorus.permissions.engine — regex-based permission matching."""

from __future__ import annotations

import pytest

from chorus.permissions.engine import (
    PRESETS,
    InvalidPermissionPatternError,
    PermissionProfile,
    PermissionResult,
    UnknownPresetError,
    check,
    format_action,
    get_preset,
)

# ---------------------------------------------------------------------------
# Core matching logic
# ---------------------------------------------------------------------------


class TestCheckAllowMatchesFirst:
    def test_allow_match_returns_allow(self) -> None:
        profile = PermissionProfile(allow=[r"tool:file:.*"], ask=[r"tool:file:.*"])
        result = check("tool:file:create /src/app.py", profile)
        assert result is PermissionResult.ALLOW

    def test_allow_checked_before_ask(self) -> None:
        profile = PermissionProfile(allow=[r"tool:bash:echo.*"], ask=[r"tool:bash:.*"])
        assert check("tool:bash:echo hello", profile) is PermissionResult.ALLOW


class TestCheckAskMatchesWhenAllowDoesNot:
    def test_ask_match_returns_ask(self) -> None:
        profile = PermissionProfile(allow=[r"tool:file:.*"], ask=[r"tool:bash:.*"])
        result = check("tool:bash:rm -rf /tmp/junk", profile)
        assert result is PermissionResult.ASK


class TestCheckDenyWhenNothingMatches:
    def test_no_match_denies(self) -> None:
        profile = PermissionProfile(allow=[r"tool:file:.*"], ask=[r"tool:bash:.*"])
        result = check("tool:self_edit:system_prompt", profile)
        assert result is PermissionResult.DENY


class TestCheckAllowTakesPriorityOverAskForSamePattern:
    def test_same_pattern_in_both_allow_wins(self) -> None:
        profile = PermissionProfile(allow=[r".*"], ask=[r".*"])
        assert check("tool:bash:anything", profile) is PermissionResult.ALLOW


class TestCheckEmptyAllowAndAskDeniesEverything:
    def test_empty_profile_denies(self) -> None:
        profile = PermissionProfile(allow=[], ask=[])
        assert check("tool:file:view README.md", profile) is PermissionResult.DENY
        assert check("tool:bash:ls", profile) is PermissionResult.DENY


class TestCheckFullActionStringFormat:
    def test_full_format_matches(self) -> None:
        profile = PermissionProfile(allow=[r"tool:bash:pip install.*"], ask=[])
        assert check("tool:bash:pip install requests", profile) is PermissionResult.ALLOW
        assert check("tool:bash:pip uninstall requests", profile) is PermissionResult.DENY


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------


class TestPresetOpen:
    def test_allows_everything(self) -> None:
        p = get_preset("open")
        assert check("tool:file:create /anything", p) is PermissionResult.ALLOW
        assert check("tool:self_edit:system_prompt", p) is PermissionResult.ALLOW

    def test_allows_bash_rm(self) -> None:
        p = get_preset("open")
        assert check("tool:bash:rm -rf /", p) is PermissionResult.ALLOW


class TestPresetStandard:
    def test_allows_file_create(self) -> None:
        p = get_preset("standard")
        assert check("tool:file:create /src/app.py", p) is PermissionResult.ALLOW

    def test_allows_file_view(self) -> None:
        p = get_preset("standard")
        assert check("tool:file:view /src/app.py", p) is PermissionResult.ALLOW

    def test_asks_bash(self) -> None:
        p = get_preset("standard")
        assert check("tool:bash:pip install requests", p) is PermissionResult.ASK

    def test_asks_git_push(self) -> None:
        p = get_preset("standard")
        assert check("tool:git:push origin main", p) is PermissionResult.ASK

    def test_allows_git_commit(self) -> None:
        p = get_preset("standard")
        assert check("tool:git:commit -m 'init'", p) is PermissionResult.ALLOW

    def test_asks_self_edit_system_prompt(self) -> None:
        p = get_preset("standard")
        assert check("tool:self_edit:system_prompt", p) is PermissionResult.ASK

    def test_allows_self_edit_docs(self) -> None:
        p = get_preset("standard")
        assert check("tool:self_edit:docs README.md", p) is PermissionResult.ALLOW

    def test_asks_self_edit_permissions(self) -> None:
        p = get_preset("standard")
        assert check("tool:self_edit:permissions open", p) is PermissionResult.ASK

    def test_asks_self_edit_model(self) -> None:
        p = get_preset("standard")
        assert check("tool:self_edit:model gpt-4o", p) is PermissionResult.ASK


class TestPresetStandardWebSearch:
    def test_standard_preset_asks_web_search(self) -> None:
        p = get_preset("standard")
        assert check("tool:web_search:enabled", p) is PermissionResult.ASK

    def test_open_preset_allows_web_search(self) -> None:
        p = get_preset("open")
        assert check("tool:web_search:enabled", p) is PermissionResult.ALLOW

    def test_locked_preset_denies_web_search(self) -> None:
        p = get_preset("locked")
        assert check("tool:web_search:enabled", p) is PermissionResult.DENY


class TestPresetLocked:
    def test_allows_file_view(self) -> None:
        p = get_preset("locked")
        assert check("tool:file:view /src/app.py", p) is PermissionResult.ALLOW

    def test_denies_file_create(self) -> None:
        p = get_preset("locked")
        assert check("tool:file:create /src/app.py", p) is PermissionResult.DENY

    def test_denies_bash(self) -> None:
        p = get_preset("locked")
        assert check("tool:bash:ls", p) is PermissionResult.DENY

    def test_denies_git_push(self) -> None:
        p = get_preset("locked")
        assert check("tool:git:push origin main", p) is PermissionResult.DENY


class TestGetPreset:
    def test_returns_correct_profile(self) -> None:
        for name in ("open", "standard", "locked"):
            p = get_preset(name)
            assert isinstance(p, PermissionProfile)
            assert p is PRESETS[name]

    def test_raises_on_unknown_name(self) -> None:
        with pytest.raises(UnknownPresetError):
            get_preset("nonexistent")


# ---------------------------------------------------------------------------
# Pattern specificity
# ---------------------------------------------------------------------------


class TestPatternSpecificity:
    def test_specific_allow_overrides_broad_ask(self) -> None:
        profile = PermissionProfile(
            allow=[r"tool:bash:echo.*"],
            ask=[r"tool:bash:.*"],
        )
        assert check("tool:bash:echo hello", profile) is PermissionResult.ALLOW
        assert check("tool:bash:rm -rf /", profile) is PermissionResult.ASK

    def test_regex_special_chars_in_action_handled(self) -> None:
        profile = PermissionProfile(
            allow=[r"tool:bash:echo \[test\]"],
            ask=[],
        )
        assert check("tool:bash:echo [test]", profile) is PermissionResult.ALLOW
        assert check("tool:bash:echo test", profile) is PermissionResult.DENY

    def test_multiline_action_string_not_matched_across_lines(self) -> None:
        profile = PermissionProfile(allow=[r"tool:bash:.*"], ask=[])
        assert check("tool:bash:echo hello\nrm -rf /", profile) is PermissionResult.DENY


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_action_string_with_colons_in_detail(self) -> None:
        profile = PermissionProfile(allow=[r"tool:bash:echo a:b:c"], ask=[])
        assert check("tool:bash:echo a:b:c", profile) is PermissionResult.ALLOW

    def test_empty_detail_string(self) -> None:
        profile = PermissionProfile(allow=[r"tool:bash:"], ask=[])
        assert check("tool:bash:", profile) is PermissionResult.ALLOW

    def test_format_action_produces_correct_string(self) -> None:
        assert format_action("bash", "pip install requests") == "tool:bash:pip install requests"
        assert format_action("file", "create /src/app.py") == "tool:file:create /src/app.py"
        assert format_action("bash", "") == "tool:bash:"
        assert format_action("git", "push origin main") == "tool:git:push origin main"

    def test_invalid_regex_raises_at_construction(self) -> None:
        with pytest.raises(InvalidPermissionPatternError):
            PermissionProfile(allow=["[invalid"], ask=[])
        with pytest.raises(InvalidPermissionPatternError):
            PermissionProfile(allow=[], ask=["(unclosed"])

    def test_profile_compiles_patterns_once(self) -> None:
        profile = PermissionProfile(allow=[r"tool:file:.*"], ask=[r"tool:bash:.*"])
        compiled_allow = profile._compiled_allow
        compiled_ask = profile._compiled_ask
        # Accessing again should return the same objects
        assert profile._compiled_allow is compiled_allow
        assert profile._compiled_ask is compiled_ask

    def test_profile_serialization_to_dict(self) -> None:
        profile = PermissionProfile(allow=[r"tool:file:.*"], ask=[r"tool:bash:.*"])
        d = profile.to_dict()
        assert d == {"allow": [r"tool:file:.*"], "ask": [r"tool:bash:.*"]}

    def test_profile_deserialization_from_dict(self) -> None:
        d = {"allow": [r"tool:file:.*"], "ask": [r"tool:bash:.*"]}
        profile = PermissionProfile.from_dict(d)
        assert profile.allow == [r"tool:file:.*"]
        assert profile.ask == [r"tool:bash:.*"]
        # Should also have compiled patterns
        assert len(profile._compiled_allow) == 1
        assert len(profile._compiled_ask) == 1
