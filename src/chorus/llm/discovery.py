"""API key validation, model discovery, and caching.

Validates Anthropic/OpenAI keys, discovers available models, and caches
results in ``~/.chorus-agents/available_models.json``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("chorus.llm.discovery")

# ---------------------------------------------------------------------------
# Known Anthropic models (API doesn't reliably list them all)
# ---------------------------------------------------------------------------

KNOWN_ANTHROPIC_MODELS: list[str] = [
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-haiku-4-20250506",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
]

# Prefixes for OpenAI chat models (public for reuse in bot.py)
OPENAI_CHAT_PREFIXES = ("gpt-", "chatgpt-", "o1-", "o3-", "o4-")


# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------


async def validate_key(
    provider_name: str,
    api_key: str,
    *,
    _client: Any = None,
) -> bool:
    """Test whether an API key is valid by making a minimal API call.

    For Anthropic: sends a tiny ``messages.create`` request.
    For OpenAI: calls ``models.list()``.

    Returns ``True`` if the key works, ``False`` otherwise.
    """
    if provider_name == "anthropic":
        return await _validate_anthropic(api_key, _client=_client)
    elif provider_name == "openai":
        return await _validate_openai(api_key, _client=_client)
    else:
        logger.warning("Unknown provider for key validation: %s", provider_name)
        return False


async def _validate_anthropic(api_key: str, *, _client: Any = None) -> bool:
    if _client is None:
        import anthropic

        _client = anthropic.AsyncAnthropic(api_key=api_key)
    try:
        await _client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True
    except Exception:
        return False


async def _validate_openai(api_key: str, *, _client: Any = None) -> bool:
    if _client is None:
        import openai

        _client = openai.AsyncOpenAI(api_key=api_key)
    try:
        await _client.models.list()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


async def discover_models(
    provider_name: str,
    api_key: str,
    *,
    _client: Any = None,
) -> list[str]:
    """Discover available models for a provider.

    Anthropic: queries the models API, falls back to known models on failure.
    OpenAI: queries the models API and filters to chat-capable models.
    """
    if provider_name == "anthropic":
        return await _discover_anthropic(api_key, _client=_client)
    elif provider_name == "openai":
        return await _discover_openai(api_key, _client=_client)
    else:
        return []


async def _discover_anthropic(api_key: str, *, _client: Any = None) -> list[str]:
    """Discover Anthropic models via the API, falling back to KNOWN_ANTHROPIC_MODELS."""
    if _client is None:
        import anthropic

        _client = anthropic.AsyncAnthropic(api_key=api_key)
    try:
        response = await _client.models.list(limit=100)
        models = sorted(m.id for m in response.data)
        if not models:
            logger.info("Anthropic models.list() returned empty — using fallback")
            return list(KNOWN_ANTHROPIC_MODELS)
        return models
    except Exception:
        logger.info("Anthropic models.list() failed — using fallback list")
        return list(KNOWN_ANTHROPIC_MODELS)


async def _discover_openai(api_key: str, *, _client: Any = None) -> list[str]:
    if _client is None:
        import openai

        _client = openai.AsyncOpenAI(api_key=api_key)
    try:
        response = await _client.models.list()
        return sorted(
            m.id for m in response.data if m.id.startswith(OPENAI_CHAT_PREFIXES)
        )
    except Exception:
        logger.exception("Failed to discover OpenAI models")
        return []


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE_FILENAME = "available_models.json"


def read_cache(chorus_home: Path) -> dict[str, Any] | None:
    """Read the cached model discovery results, or ``None`` if missing."""
    cache_path = chorus_home / _CACHE_FILENAME
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text())  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError):
        return None


def write_cache(chorus_home: Path, data: dict[str, Any]) -> None:
    """Write model discovery results to the cache file."""
    chorus_home.mkdir(parents=True, exist_ok=True)
    cache_path = chorus_home / _CACHE_FILENAME
    cache_path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def get_cached_models(chorus_home: Path) -> list[str]:
    """Read the cache and return a flat, sorted, deduplicated list of all model names."""
    cache = read_cache(chorus_home)
    if cache is None:
        return []
    providers = cache.get("providers", {})
    models: set[str] = set()
    for info in providers.values():
        if isinstance(info, dict) and info.get("valid"):
            for m in info.get("models", []):
                models.add(m)
    return sorted(models)


# ---------------------------------------------------------------------------
# Combined validate + discover
# ---------------------------------------------------------------------------


async def validate_and_discover(
    chorus_home: Path,
    anthropic_key: str | None = None,
    openai_key: str | None = None,
    *,
    _anthropic_client: Any = None,
    _openai_client: Any = None,
) -> dict[str, Any]:
    """Validate keys, discover models, write cache, and return results."""
    providers: dict[str, Any] = {}

    if anthropic_key:
        valid = await validate_key(
            "anthropic", anthropic_key, _client=_anthropic_client
        )
        models = (
            await discover_models("anthropic", anthropic_key, _client=_anthropic_client)
            if valid
            else []
        )
        providers["anthropic"] = {"valid": valid, "models": models}

    if openai_key:
        valid = await validate_key("openai", openai_key, _client=_openai_client)
        models = (
            await discover_models("openai", openai_key, _client=_openai_client)
            if valid
            else []
        )
        providers["openai"] = {"valid": valid, "models": models}

    result: dict[str, Any] = {
        "last_updated": datetime.now(UTC).isoformat(),
        "providers": providers,
    }

    write_cache(chorus_home, result)
    return result
