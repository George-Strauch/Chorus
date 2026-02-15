"""Tests for chorus.config — bot configuration and environment variables."""

import json
from pathlib import Path

import pytest

from chorus.config import BotConfig, GlobalConfig, validate_model_available


class TestConfig:
    def test_loads_from_env(self, tmp_env: BotConfig) -> None:
        config = BotConfig.from_env()
        assert config.discord_token == "test-token-123"
        assert config.anthropic_api_key == "sk-ant-test"
        assert config.openai_api_key == "sk-openai-test"
        assert config.dev_guild_id == 999888777

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISCORD_TOKEN", "tok")
        # Clear optional keys so they are absent
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CHORUS_HOME", raising=False)
        monkeypatch.delenv("DEV_GUILD_ID", raising=False)
        config = BotConfig.from_env()
        assert config.anthropic_api_key is None
        assert config.openai_api_key is None
        assert config.dev_guild_id is None
        assert config.chorus_home == Path.home() / ".chorus-agents"

    def test_chorus_home_path_expansion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISCORD_TOKEN", "tok")
        monkeypatch.setenv("CHORUS_HOME", "~/my-chorus")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DEV_GUILD_ID", raising=False)
        config = BotConfig.from_env()
        assert config.chorus_home.is_absolute()
        assert "~" not in str(config.chorus_home)

    def test_missing_discord_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DISCORD_TOKEN", raising=False)
        with pytest.raises(ValueError, match="DISCORD_TOKEN"):
            BotConfig.from_env()

    def test_empty_string_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISCORD_TOKEN", "   ")
        with pytest.raises(ValueError, match="DISCORD_TOKEN"):
            BotConfig.from_env()

    def test_strips_whitespace_from_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISCORD_TOKEN", "  my-token  ")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CHORUS_HOME", raising=False)
        monkeypatch.delenv("DEV_GUILD_ID", raising=False)
        config = BotConfig.from_env()
        assert config.discord_token == "my-token"

    def test_live_test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISCORD_TOKEN", "tok")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CHORUS_HOME", raising=False)
        monkeypatch.delenv("DEV_GUILD_ID", raising=False)
        monkeypatch.delenv("LIVE_TEST_ENABLED", raising=False)
        monkeypatch.delenv("LIVE_TEST_CHANNEL", raising=False)
        config = BotConfig.from_env()
        assert config.live_test_enabled is False
        assert config.live_test_channel is None

    def test_live_test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISCORD_TOKEN", "tok")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CHORUS_HOME", raising=False)
        monkeypatch.delenv("DEV_GUILD_ID", raising=False)
        monkeypatch.setenv("LIVE_TEST_ENABLED", "true")
        monkeypatch.setenv("LIVE_TEST_CHANNEL", "my-test-chan")
        config = BotConfig.from_env()
        assert config.live_test_enabled is True
        assert config.live_test_channel == "my-test-chan"

    def test_live_test_enabled_variants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISCORD_TOKEN", "tok")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CHORUS_HOME", raising=False)
        monkeypatch.delenv("DEV_GUILD_ID", raising=False)
        monkeypatch.delenv("LIVE_TEST_CHANNEL", raising=False)
        for val in ("1", "yes", "True", "YES"):
            monkeypatch.setenv("LIVE_TEST_ENABLED", val)
            config = BotConfig.from_env()
            assert config.live_test_enabled is True, f"Failed for LIVE_TEST_ENABLED={val}"

        for val in ("0", "false", "no", ""):
            monkeypatch.setenv("LIVE_TEST_ENABLED", val)
            config = BotConfig.from_env()
            assert config.live_test_enabled is False, f"Failed for LIVE_TEST_ENABLED={val}"

    def test_scope_path_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISCORD_TOKEN", "tok")
        monkeypatch.setenv("CHORUS_SCOPE_PATH", "/mnt/host")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CHORUS_HOME", raising=False)
        monkeypatch.delenv("DEV_GUILD_ID", raising=False)
        config = BotConfig.from_env()
        assert config.scope_path == Path("/mnt/host")
        assert config.scope_path.is_absolute()

    def test_scope_path_defaults_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISCORD_TOKEN", "tok")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CHORUS_HOME", raising=False)
        monkeypatch.delenv("DEV_GUILD_ID", raising=False)
        monkeypatch.delenv("CHORUS_SCOPE_PATH", raising=False)
        config = BotConfig.from_env()
        assert config.scope_path is None

    def test_scope_path_resolves_relative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISCORD_TOKEN", "tok")
        monkeypatch.setenv("CHORUS_SCOPE_PATH", "~/host-files")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CHORUS_HOME", raising=False)
        monkeypatch.delenv("DEV_GUILD_ID", raising=False)
        config = BotConfig.from_env()
        assert config.scope_path is not None
        assert config.scope_path.is_absolute()
        assert "~" not in str(config.scope_path)

    def test_at_least_one_llm_key_required(self) -> None:
        pytest.skip("Not implemented yet — TODO 008")


class TestGlobalConfig:
    def test_global_config_loads_from_json(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "default_model": "claude-sonnet-4-20250514",
            "default_permissions": "open",
            "idle_timeout": 3600,
            "max_tool_loop_iterations": 50,
            "max_bash_timeout": 300,
        }))
        cfg = GlobalConfig.load(config_path)
        assert cfg.default_model == "claude-sonnet-4-20250514"
        assert cfg.default_permissions == "open"
        assert cfg.idle_timeout == 3600
        assert cfg.max_tool_loop_iterations == 50
        assert cfg.max_bash_timeout == 300

    def test_global_config_creates_default_on_missing(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        assert not config_path.exists()
        cfg = GlobalConfig.load(config_path)
        assert config_path.exists()
        assert cfg.default_model is None
        assert cfg.default_permissions == "standard"

    def test_global_config_default_model_is_none_until_set(self, tmp_path: Path) -> None:
        cfg = GlobalConfig.load(tmp_path / "config.json")
        assert cfg.default_model is None

    def test_global_config_default_permissions_is_standard(self, tmp_path: Path) -> None:
        cfg = GlobalConfig.load(tmp_path / "config.json")
        assert cfg.default_permissions == "standard"

    def test_global_config_write_through_persists(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        cfg = GlobalConfig.load(config_path)
        cfg.default_model = "gpt-4o"
        cfg.save(config_path)

        cfg2 = GlobalConfig.load(config_path)
        assert cfg2.default_model == "gpt-4o"

    def test_global_config_validates_model_against_available(self, tmp_path: Path) -> None:
        cache = {
            "providers": {
                "anthropic": {"valid": True, "models": ["claude-sonnet-4-20250514"]},
            }
        }
        (tmp_path / "available_models.json").write_text(json.dumps(cache))
        assert validate_model_available("claude-sonnet-4-20250514", tmp_path) is True

    def test_global_config_rejects_unknown_model(self, tmp_path: Path) -> None:
        cache = {
            "providers": {
                "anthropic": {"valid": True, "models": ["claude-sonnet-4-20250514"]},
            }
        }
        (tmp_path / "available_models.json").write_text(json.dumps(cache))
        assert validate_model_available("nonexistent-model", tmp_path) is False

    def test_global_config_idle_timeout_default(self, tmp_path: Path) -> None:
        cfg = GlobalConfig.load(tmp_path / "config.json")
        assert cfg.idle_timeout == 1800

    def test_global_config_max_iterations_default_is_40(self, tmp_path: Path) -> None:
        cfg = GlobalConfig.load(tmp_path / "config.json")
        assert cfg.max_tool_loop_iterations == 40

    def test_global_config_serialization_roundtrip(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        cfg = GlobalConfig(
            default_model="gpt-4o",
            default_permissions="open",
            idle_timeout=900,
            max_tool_loop_iterations=10,
            max_bash_timeout=60,
        )
        cfg.save(config_path)

        cfg2 = GlobalConfig.load(config_path)
        assert cfg2.default_model == cfg.default_model
        assert cfg2.default_permissions == cfg.default_permissions
        assert cfg2.idle_timeout == cfg.idle_timeout
        assert cfg2.max_tool_loop_iterations == cfg.max_tool_loop_iterations
        assert cfg2.max_bash_timeout == cfg.max_bash_timeout
