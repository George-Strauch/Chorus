"""Tests for chorus.config — bot configuration and environment variables."""

from pathlib import Path

import pytest

from chorus.config import BotConfig


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

    def test_at_least_one_llm_key_required(self) -> None:
        pytest.skip("Not implemented yet — TODO 008")
