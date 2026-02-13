"""Tests for chorus.llm.discovery — key validation and model discovery."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from chorus.llm.discovery import (
    discover_models,
    read_cache,
    validate_and_discover,
    validate_key,
    write_cache,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------


class TestKeyValidation:
    @pytest.mark.asyncio
    async def test_validate_key_anthropic_valid(self) -> None:
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=MagicMock())

        result = await validate_key("anthropic", "sk-ant-test", _client=mock_client)
        assert result is True
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_key_anthropic_invalid(self) -> None:
        import anthropic

        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.AuthenticationError(
                message="Invalid key",
                response=mock_response,
                body=None,
            )
        )

        result = await validate_key("anthropic", "sk-ant-invalid", _client=mock_client)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_key_openai_valid(self) -> None:
        mock_client = MagicMock()
        mock_client.models = MagicMock()
        mock_client.models.list = AsyncMock(return_value=MagicMock())

        result = await validate_key("openai", "sk-test", _client=mock_client)
        assert result is True
        mock_client.models.list.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_key_openai_invalid(self) -> None:
        import openai

        mock_client = MagicMock()
        mock_client.models = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_response.json.return_value = {"error": {"message": "Invalid key"}}
        mock_client.models.list = AsyncMock(
            side_effect=openai.AuthenticationError(
                message="Invalid key",
                response=mock_response,
                body=None,
            )
        )

        result = await validate_key("openai", "sk-invalid", _client=mock_client)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_key_unknown_provider(self) -> None:
        result = await validate_key("unknown_provider", "some-key")
        assert result is False


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


class TestModelDiscovery:
    @pytest.mark.asyncio
    async def test_discover_models_anthropic_returns_known_models(self) -> None:
        models = await discover_models("anthropic", "sk-ant-test")
        assert isinstance(models, list)
        assert len(models) > 0
        # Should contain well-known Claude models
        assert any("claude" in m for m in models)

    @pytest.mark.asyncio
    async def test_discover_models_openai_filters_chat_models(self) -> None:
        model1 = MagicMock()
        model1.id = "gpt-4o"
        model2 = MagicMock()
        model2.id = "gpt-4o-mini"
        model3 = MagicMock()
        model3.id = "dall-e-3"
        model4 = MagicMock()
        model4.id = "text-embedding-ada-002"
        model5 = MagicMock()
        model5.id = "chatgpt-4o-latest"

        mock_client = MagicMock()
        mock_client.models = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [model1, model2, model3, model4, model5]
        mock_client.models.list = AsyncMock(return_value=mock_response)

        models = await discover_models("openai", "sk-test", _client=mock_client)
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "chatgpt-4o-latest" in models
        # Image/embedding models should be filtered out
        assert "dall-e-3" not in models
        assert "text-embedding-ada-002" not in models

    @pytest.mark.asyncio
    async def test_discover_models_unknown_provider(self) -> None:
        models = await discover_models("unknown", "some-key")
        assert models == []


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class TestCache:
    def test_write_cache(self, tmp_path: Path) -> None:
        data = {
            "last_updated": "2026-02-12T10:00:00Z",
            "providers": {
                "anthropic": {"valid": True, "models": ["claude-sonnet-4-20250514"]},
            },
        }
        write_cache(tmp_path, data)
        cache_file = tmp_path / "available_models.json"
        assert cache_file.exists()
        loaded = json.loads(cache_file.read_text())
        assert loaded == data

    def test_read_cache(self, tmp_path: Path) -> None:
        data = {
            "last_updated": "2026-02-12T10:00:00Z",
            "providers": {"anthropic": {"valid": True, "models": ["claude-sonnet-4-20250514"]}},
        }
        cache_file = tmp_path / "available_models.json"
        cache_file.write_text(json.dumps(data))

        result = read_cache(tmp_path)
        assert result == data

    def test_read_cache_missing_file(self, tmp_path: Path) -> None:
        result = read_cache(tmp_path)
        assert result is None

    def test_cache_refresh_overwrites(self, tmp_path: Path) -> None:
        old = {"last_updated": "old", "providers": {}}
        new = {"last_updated": "new", "providers": {"anthropic": {"valid": True, "models": []}}}
        write_cache(tmp_path, old)
        write_cache(tmp_path, new)
        result = read_cache(tmp_path)
        assert result is not None
        assert result["last_updated"] == "new"


# ---------------------------------------------------------------------------
# validate_and_discover
# ---------------------------------------------------------------------------


class TestValidateAndDiscover:
    @pytest.mark.asyncio
    async def test_validate_and_discover_writes_cache(self, tmp_path: Path) -> None:
        mock_anth_client = MagicMock()
        mock_anth_client.messages = MagicMock()
        mock_anth_client.messages.create = AsyncMock(return_value=MagicMock())

        result = await validate_and_discover(
            tmp_path,
            anthropic_key="sk-ant-test",
            _anthropic_client=mock_anth_client,
        )

        assert result["providers"]["anthropic"]["valid"] is True
        assert len(result["providers"]["anthropic"]["models"]) > 0
        assert "last_updated" in result

        # Should have written cache
        cached = read_cache(tmp_path)
        assert cached is not None
        assert cached["providers"]["anthropic"]["valid"] is True

    @pytest.mark.asyncio
    async def test_validate_and_discover_no_keys(self, tmp_path: Path) -> None:
        result = await validate_and_discover(tmp_path)
        # No providers should have valid keys
        assert result["providers"] == {}


# ---------------------------------------------------------------------------
# Integration: validate_and_discover cache file and per-provider status
# ---------------------------------------------------------------------------


class TestValidateAndDiscoverIntegration:
    @pytest.mark.asyncio
    async def test_validate_keys_command_updates_cache_file(self, tmp_path: Path) -> None:
        """Calling validate_and_discover writes a cache file that can be read back."""
        mock_anth_client = MagicMock()
        mock_anth_client.messages = MagicMock()
        mock_anth_client.messages.create = AsyncMock(return_value=MagicMock())

        await validate_and_discover(
            tmp_path,
            anthropic_key="sk-ant-test",
            _anthropic_client=mock_anth_client,
        )

        cache_file = tmp_path / "available_models.json"
        assert cache_file.exists()
        cached = json.loads(cache_file.read_text())
        assert "providers" in cached
        assert "anthropic" in cached["providers"]
        assert cached["providers"]["anthropic"]["valid"] is True
        assert len(cached["providers"]["anthropic"]["models"]) > 0

    @pytest.mark.asyncio
    async def test_validate_keys_reports_per_provider_status(self, tmp_path: Path) -> None:
        """With both keys, result has separate status for each provider."""
        mock_anth_client = MagicMock()
        mock_anth_client.messages = MagicMock()
        mock_anth_client.messages.create = AsyncMock(return_value=MagicMock())

        mock_oai_client = MagicMock()
        mock_oai_client.models = MagicMock()
        model1 = MagicMock()
        model1.id = "gpt-4o"
        mock_response = MagicMock()
        mock_response.data = [model1]
        mock_oai_client.models.list = AsyncMock(return_value=mock_response)

        result = await validate_and_discover(
            tmp_path,
            anthropic_key="sk-ant-test",
            openai_key="sk-oai-test",
            _anthropic_client=mock_anth_client,
            _openai_client=mock_oai_client,
        )

        assert "anthropic" in result["providers"]
        assert "openai" in result["providers"]
        assert result["providers"]["anthropic"]["valid"] is True
        assert result["providers"]["openai"]["valid"] is True
        assert "gpt-4o" in result["providers"]["openai"]["models"]
