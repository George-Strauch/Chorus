"""Information tools — model listing and agent self-awareness."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


async def list_models(*, chorus_home: Path | None = None) -> str:
    """Return a JSON list of available models from the discovery cache."""
    if chorus_home is None:
        return json.dumps({"error": "No chorus_home configured"})

    from chorus.llm.discovery import get_cached_models

    models = get_cached_models(chorus_home)
    if not models:
        return json.dumps({
            "models": [],
            "message": "No models cached. Run /settings validate-keys to discover models.",
        })

    return json.dumps({"models": models, "count": len(models)})
